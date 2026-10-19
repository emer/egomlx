// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package egomlx

import (
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"strconv"
	"strings"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/base/iox/jsonx"
	"cogentcore.org/core/math32"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/paths"
	"github.com/emer/emergent/v2/weights"
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/nn"
)

// Scope is an alias for a [gomlx] model Scope,
// which holds the weight variables for the model.
//
//	model = github.com/gomlx/gomlx/ml/model
type Scope = model.Scope

// Rand is the random number generator interface used for
// reproducible weight initialization. The [Network] [Rand]
// field (a randx.Rand) satisfies this interface.
type Rand interface {
	// Float64 returns a pseudo-random number in [0.0, 1.0).
	Float64() float64

	// NormFloat64 returns a normally distributed number
	// with mean 0 and standard deviation 1.
	NormFloat64() float64
}

// globalRand provides a fallback random source using the
// math/rand/v2 global source, for when no network generator
// is available.
type globalRand struct{}

func (globalRand) Float64() float64     { return rand.Float64() }
func (globalRand) NormFloat64() float64 { return rand.NormFloat64() }

var randFallback = globalRand{}

// Path implements an emergent neural network pathway,
// which connects a sending [Layer] to a receiving [Layer],
// using a dense (linear) transformation of the sending layer's
// activation values, with a bias term.
//
// The path type ([PathTypes]) is stored in [Params], following
// the leabra and axon convention.
//
// The pathway is the owner of the weight variables of the dense
// transformation: the weights have shape [NSend, NRecv], with the
// sending units as the outer-most dimension, corresponding to the
// [compute.DenseLayoutInputOutputs] weight layout in [gomlx].
// This means that the synapse values are ordered sender-based:
// for each sending unit, the full set of receiving units.
type Path struct {
	emer.PathBase

	// Params contains all of the path parameters.
	Params PathParams `display:"inline"`

	// Send is the sending layer for this pathway.
	Send *Layer

	// Recv is the receiving layer for this pathway.
	Recv *Layer

	// Kernel is the [gomlx] variable for the dense transformation
	// weights, with shape [NSend, NRecv].
	// It is created when the network is built.
	Kernel *model.Variable `display:"-"`

	// Bias is the [gomlx] variable for the dense transformation
	// bias term, with shape [NRecv].
	// It is created when the network is built.
	Bias *model.Variable `display:"-"`

	// KernelVals is the read-back of the [Kernel] variable values,
	// in [NSend][NRecv] sender-based order -- the same order as
	// the synapse values below.
	// It is updated after each forward pass and weight change.
	KernelVals []float32 `display:"-"`

	// BiasVals is the read-back of the [Bias] variable values,
	// in [NRecv] order.
	BiasVals []float32 `display:"-"`

	// KernelDWt is the read-back of the weight-change (gradient)
	// values for the [Kernel], computed during the [Network] Backward
	// pass, in [NSend][NRecv] sender-based order.
	// Only updated if DWt readback was requested: see [DWtTypes].
	KernelDWt []float32 `display:"-"`

	// BiasDWt is the read-back of the weight-change (gradient) values
	// for the [Bias], in [NRecv] order.
	BiasDWt []float32 `display:"-"`
}

// emer.Path interface methods

func (pt *Path) TypeName() string      { return pt.Params.Type.String() }
func (pt *Path) TypeNumber() int       { return int(pt.Params.Type) }
func (pt *Path) StyleObject() any      { return pt }
func (pt *Path) SendLayer() emer.Layer { return pt.Send }
func (pt *Path) RecvLayer() emer.Layer { return pt.Recv }

// NumSyns returns the number of synapses for this path,
// which is the number of send units times the number of recv units
// for the full connectivity pattern used by this toolkit.
func (pt *Path) NumSyns() int {
	return pt.Send.Shape.Len() * pt.Recv.Shape.Len()
}

// SynIndex returns the index of the synapse between the given
// sending and receiving unit indexes (1D, flat indexes),
// in the sender-based order: sending units outer-most.
// Returns -1 if the indexes are invalid.
func (pt *Path) SynIndex(sidx, ridx int) int {
	nSend := pt.Send.Shape.Len()
	nRecv := pt.Recv.Shape.Len()
	if sidx < 0 || sidx >= nSend || ridx < 0 || ridx >= nRecv {
		return -1
	}
	return sidx*nRecv + ridx
}

// SynVarNames returns the names of all the variables on the synapse.
func (pt *Path) SynVarNames() []string {
	return SynapseVars
}

// SynVarNum returns the number of synapse-level variables.
func (pt *Path) SynVarNum() int {
	return SynapseVarN
}

// SynVarIndex returns the index of the given variable name within
// the synapse variable list, or -1 and error message if not found.
func (pt *Path) SynVarIndex(varNm string) (int, error) {
	return SynapseVarIndexByName(varNm)
}

// SynValue1D returns the value of the given variable index
// on the given synapse index.
// Returns NaN on invalid index.
// The Wt variable reads from the [KernelVals] readback buffer,
// and the DWt variable reads from the [KernelDWt] gradient buffer,
// which only has values after a Backward pass with DWt readback.
func (pt *Path) SynValue1D(varIndex int, synIndex int) float32 {
	if varIndex < 0 || varIndex >= pt.SynVarNum() {
		return math32.NaN()
	}
	vals := pt.kernelVarVals(varIndex)
	if synIndex < 0 || synIndex >= len(vals) {
		return math32.NaN()
	}
	return vals[synIndex]
}

// kernelVarVals returns the readback buffer for the given synapse
// variable index: Wt -> KernelVals, DWt -> KernelDWt.
// Returns nil for unsupported or unpopulated variables.
func (pt *Path) kernelVarVals(varIndex int) []float32 {
	if varIndex < 0 || varIndex >= SynapseVarN {
		return nil
	}
	switch SynapseVars[varIndex] {
	case "Wt":
		return pt.KernelVals
	case "DWt":
		return pt.KernelDWt
	}
	return nil
}

// SynValues fills in the values of the given variable name for each synapse,
// using the natural sender-based ordering of the synapses,
// into the given float32 slice (only resized if not big enough).
// Returns an error on invalid variable name.
// Wt values come from [KernelVals], and DWt from [KernelDWt],
// which is NaN-filled until a Backward pass with DWt readback occurs.
func (pt *Path) SynValues(vals *[]float32, varNm string) error {
	vi, err := pt.SynVarIndex(varNm)
	if err != nil {
		return errors.Log(err)
	}
	ns := pt.NumSyns()
	if *vals == nil || cap(*vals) < ns {
		*vals = make([]float32, ns)
	} else if len(*vals) < ns {
		*vals = (*vals)[:ns]
	}
	src := pt.kernelVarVals(vi)
	if len(src) == 0 {
		nan := math32.NaN()
		for i := range *vals {
			(*vals)[i] = nan
		}
		return nil
	}
	copy(*vals, src)
	return nil
}

// ParamsString returns a listing of all parameters in the pathway.
// If nonDefault is true, only report those not at their default values.
func (pt *Path) ParamsString(nonDefault bool) string {
	var b strings.Builder
	b.WriteString("  ////////  Path: " + pt.Name + "\n")
	b.WriteString(pt.Params.ParamsString(nonDefault))
	return b.String()
}

// Defaults sets the default values for the path parameters.
func (pt *Path) Defaults() {
	pt.Params.Path = pt
	pt.Params.Defaults()
}

// UpdateParams updates all parameters given any changes that might
// have been made to individual values, and rebuilds the network
// model if any graph-affecting parameters (e.g., bias) changed.
func (pt *Path) UpdateParams() {
	pt.Params.Update()
	if pt.Recv != nil && pt.Recv.Network != nil {
		pt.Recv.Network.UpdateParams()
	}
}

// Connect sets the sending and receiving layers, the connectivity
// pattern, and the path type, and sets the standard path name
// of SendLayer.Name + "To" + RecvLayer.Name.
func (pt *Path) Connect(slay, rlay *Layer, pat paths.Pattern, typ PathTypes) {
	pt.Send = slay
	pt.Recv = rlay
	pt.Pattern = pat
	pt.Params.Type = typ
	pt.Name = slay.Name + "To" + rlay.Name
	pt.Params.Path = pt
}

// allocBuffers allocates the weight readback buffers
// for the [NSend][NRecv] kernel and the [NRecv] bias,
// along with the DWt gradient buffers.
func (pt *Path) allocBuffers() {
	nSend := pt.Send.Shape.Len()
	nRecv := pt.Recv.Shape.Len()
	if cap(pt.KernelVals) < nSend*nRecv {
		pt.KernelVals = make([]float32, nSend*nRecv)
	} else {
		pt.KernelVals = pt.KernelVals[:nSend*nRecv]
	}
	if cap(pt.BiasVals) < nRecv {
		pt.BiasVals = make([]float32, nRecv)
	} else {
		pt.BiasVals = pt.BiasVals[:nRecv]
	}
	if cap(pt.KernelDWt) < nSend*nRecv {
		pt.KernelDWt = make([]float32, nSend*nRecv)
	} else {
		pt.KernelDWt = pt.KernelDWt[:nSend*nRecv]
	}
	if cap(pt.BiasDWt) < nRecv {
		pt.BiasDWt = make([]float32, nRecv)
	} else {
		pt.BiasDWt = pt.BiasDWt[:nRecv]
	}
}

// learnable returns true if this pathway can learn during the
// Backward pass: it is not Off, and [PathParams.Learn] is On.
func (pt *Path) learnable() bool {
	return !pt.Off && pt.Params.Learn
}

// zeroDWt zeros the DWt gradient readback buffers.
func (pt *Path) zeroDWt() {
	clear(pt.KernelDWt)
	clear(pt.BiasDWt)
}

// initWeights initializes the [Kernel] and [Bias] variable values
// using the [WtInitParams] settings, and the network random number
// generator, for reproducible weight initialization per seed.
// He initialization uses a normal distribution with standard deviation
// sqrt(1 / fanIn), Glorot uses sqrt(1 / (fanIn + fanOut)), etc.
func (pt *Path) InitWeights() {
	nSend := pt.Send.Shape.Len()
	nRecv := pt.Recv.Shape.Len()
	rand := pt.rand()
	wip := &pt.Params.WtInit
	pt.zeroDWt()

	fanIn := float64(nSend)
	fanOut := float64(nRecv)
	limit := func() float64 {
		switch wip.Init {
		case He:
			return math.Sqrt(1.0/fanIn) * wip.Scale
		case Glorot:
			return math.Sqrt(1.0/(fanIn+fanOut)) * wip.Scale
		case Uniform:
			return math.Sqrt(3.0/(fanIn+fanOut)) * wip.Scale
		default:
			return wip.Scale
		}
	}()

	kv := make([]float32, nSend*nRecv)
	for i := range kv {
		switch wip.Init {
		case He, Glorot:
			kv[i] = float32(rand.NormFloat64()*limit + wip.Mean)
		case Normal:
			kv[i] = float32(rand.NormFloat64()*wip.Scale + wip.Mean)
		case TruncNormal:
			for {
				vl := rand.NormFloat64()*2*wip.Scale + wip.Mean
				if vl > wip.Mean-2*wip.Scale && vl < wip.Mean+2*wip.Scale {
					kv[i] = float32(vl)
					break
				}
			}
		case Uniform:
			kv[i] = float32((rand.Float64()*2 - 1) * limit)
		case Zero:
			kv[i] = 0
		}
	}

	if pt.Kernel != nil {
		err := pt.Kernel.SetValue(tensors.FromFlatDataAndDimensions(kv, nSend, nRecv))
		if err != nil {
			errors.Log(fmt.Errorf("Path %s: InitWeights: %w", pt.Name, err))
		}
		pt.readbackKernel()
	}

	if pt.Bias != nil && pt.Params.UseBias {
		bv := make([]float32, nRecv)
		err := pt.Bias.SetValue(tensors.FromFlatDataAndDimensions(bv, nRecv))
		if err != nil {
			errors.Log(fmt.Errorf("Path %s: InitWeights: bias: %w", pt.Name, err))
		}
		pt.readbackBias()
	}
}

// rand returns the network random number generator to use
// for reproducible initialization, or a global fallback.
func (pt *Path) rand() Rand {
	if pt.Recv != nil && pt.Recv.Network != nil && pt.Recv.Network.Rand != nil {
		return pt.Recv.Network.Rand
	}
	return randFallback
}

// readbackKernel reads back the [Kernel] variable values into
// the [KernelVals] buffer.
func (pt *Path) readbackKernel() {
	if pt.Kernel == nil || !pt.Kernel.HasValue() {
		return
	}
	kv := tensors.MustCopyFlatData[float32](pt.Kernel.MustValue())
	n := len(pt.KernelVals)
	if len(kv) != n {
		return
	}
	copy(pt.KernelVals, kv)
}

// readbackBias reads back the [Bias] variable values into
// the [BiasVals] buffer.
func (pt *Path) readbackBias() {
	if pt.Bias == nil || !pt.Bias.HasValue() {
		return
	}
	bv := tensors.MustCopyFlatData[float32](pt.Bias.MustValue())
	n := len(pt.BiasVals)
	if len(bv) != n {
		return
	}
	copy(pt.BiasVals, bv)
}

// ReadbackWeights reads back the current weight variable values
// into the readback buffers, so they are available for display
// and other access without re-running the model.
func (pt *Path) ReadbackWeights() {
	pt.readbackKernel()
	pt.readbackBias()
}

// SetWtValue sets a single weight value in the [Kernel] variable
// for the given sending and receiving unit indexes, and updates
// the readback buffer.
func (pt *Path) SetWtValue(sidx, ridx int, val float32) error {
	syi := pt.SynIndex(sidx, ridx)
	if syi < 0 {
		return fmt.Errorf("Path %s: SetWtValue: no synapse between send %d and recv %d", pt.Name, sidx, ridx)
	}
	if syi >= len(pt.KernelVals) {
		return fmt.Errorf("Path %s: SetWtValue: synapse index %d out of range", pt.Name, syi)
	}
	pt.KernelVals[syi] = val
	nSend := pt.Send.Shape.Len()
	nRecv := pt.Recv.Shape.Len()
	if pt.Kernel == nil {
		return nil
	}
	err := pt.Kernel.SetValue(tensors.FromFlatDataAndDimensions(pt.KernelVals, nSend, nRecv))
	if err != nil {
		return errors.Log(err)
	}
	return nil
}

// buildDense builds the [gomlx] computation graph for the dense
// transformation of this pathway, creating the [Kernel] and [Bias]
// variables in a scope named for the path, and returning the
// linear (pre-activation) output node.
// scope is the root model scope, and g is the graph to build on.
func (pt *Path) buildDense(scope *Scope, g *graph.Graph) (*graph.Node, error) {
	dscope := scope.In("%s", pt.Name).In("dense")

	nSend := pt.Send.Shape.Len()
	nRecv := pt.Recv.Shape.Len()

	kernelShp := shapes.Make(dtypes.Float32, nSend, nRecv)
	pt.Kernel = dscope.VariableWithShape("weights", kernelShp)

	pt.Bias = dscope.VariableWithShape("bias", shapes.Make(dtypes.Float32, nRecv))

	kNode := pt.Kernel.NodeValue(g)
	var bNode *graph.Node
	if pt.Params.UseBias {
		bNode = pt.Bias.NodeValue(g)
	}
	if pt.Send.Act == nil {
		return nil, fmt.Errorf("Path %s: sending layer %s has no activation node -- build it first", pt.Name, pt.Send.Name)
	}
	out := nn.Dense(pt.Send.Act, kNode, bNode, compute.DenseLayoutInputOutputs)
	pt.allocBuffers()
	pt.ReadbackWeights()
	return out, nil
}

///////////////////////////////////////////////////////////////////////
//  Weights File

// WriteWeightsJSON writes the weights from this pathway from the
// receiver-side perspective in a JSON text format, using the
// [weights.Path] structure and standard JSON output.
// The weights are written from the [KernelVals] readback buffer,
// with receiving units as the outer loop (the standard
// receiver-side perspective of the weights format).
func (pt *Path) WriteWeightsJSON(w io.Writer, depth int) {
	nSend := pt.Send.Shape.Len()
	nRecv := pt.Recv.Shape.Len()
	wpt := weights.Path{From: pt.Send.Name}
	if pt.Params.UseBias {
		wpt.SetMetaData("UseBias", "true")
	}
	wpt.Rs = make([]weights.Recv, 0, nRecv)
	for ri := 0; ri < nRecv; ri++ {
		rcv := weights.Recv{Ri: ri, N: nSend, Si: make([]int, 0, nSend)}
		for si := 0; si < nSend; si++ {
			rcv.Si = append(rcv.Si, si)
			rcv.Wt = append(rcv.Wt, pt.SynValue1D(0, si*nRecv+ri))
		}
		wpt.Rs = append(wpt.Rs, rcv)
	}
	_ = depth // indentation handled by JSON output
	errors.Log(jsonx.Write(&wpt, w))
}

// SetWeights sets the weights for this pathway from the given
// [weights.Path] decoded values, writing them into the [Kernel]
// variable and updating the readback buffers.
func (pt *Path) SetWeights(pw *weights.Path) error {
	if pw.MetaData != nil {
		if ub, ok := pw.MetaData["UseBias"]; ok {
			pv, _ := strconv.ParseBool(ub)
			pt.Params.UseBias = pv
		}
	}
	nSend := pt.Send.Shape.Len()
	nRecv := pt.Recv.Shape.Len()
	var err error
	for i := range pw.Rs {
		pr := &pw.Rs[i]
		for si := range pr.Si {
			syi := pr.Si[si]*nRecv + pr.Ri
			if syi < 0 || syi >= len(pt.KernelVals) {
				err = fmt.Errorf("Path %s: SetWeights: synapse index %d out of range", pt.Name, syi)
				break
			}
			pt.KernelVals[syi] = pr.Wt[si]
		}
	}
	if err != nil {
		return errors.Log(err)
	}
	if pt.Kernel != nil {
		err := pt.Kernel.SetValue(tensors.FromFlatDataAndDimensions(pt.KernelVals, nSend, nRecv))
		if err != nil {
			return errors.Log(err)
		}
		pt.ReadbackWeights()
	}
	return nil
}
