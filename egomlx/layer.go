// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package egomlx

import (
	"fmt"
	"io"
	"strings"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/math32"
	"cogentcore.org/lab/tensor"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/weights"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/layers/activation"
)

// Layer implements an emergent neural network layer backed by a [gomlx] model.
//
// An egomlx [Layer] corresponds to a named set of units in the model.
// Input layers have no transformation of their own: their Act values
// are the external input patterns applied via [ApplyExt].
// All other layers apply a dense (linear) transformation of the
// activations of all their sending layers -- i.e., the contributions of
// all their receiving paths are summed -- and then a nonlinear activation
// function, to compute their Act values from their Net values.
type Layer struct {
	emer.LayerBase

	// Network is a back-reference to the parent network,
	// set when the layer is added by the network.
	Network *Network `display:"-"`

	// Type is the type of the layer (Input, Hidden, Target).
	Type LayerTypes `display:"inline"`

	// Params contains all of the layer parameters.
	Params LayerParams `display:"inline"`

	// RecvPaths are the receiving pathways into this layer from other layers.
	RecvPaths []*Path

	// SendPaths are the sending pathways from this layer to other layers.
	SendPaths []*Path

	// NeuronVars holds the unit variable values in
	// [NData][NUnits][NVars] order -- the data-parallel ndata
	// index is the outer-most dimension, and the variable index
	// (see [NeuronVars]) is the inner-most.
	// Allocated by the network Build method.
	NeuronVars []float32 `display:"-"`

	// Input is the graph input node for an input layer -- the
	// runtime values of the layer's Act, with no transformation.
	// Nil for non-input layers.
	Input *graph.Node `display:"-"`

	// Net is the pre-activation (net input) output node of this layer.
	// For input layers, this is the same as the Input node.
	// Nil before the network is built.
	Net *graph.Node `display:"-"`

	// Act is the activation output node of this layer -- the
	// final computed values of the layer.
	// For input layers, this is the same as the Input node.
	// Nil before the network is built.
	Act *graph.Node `display:"-"`
}

// emer.Layer interface methods

func (ly *Layer) TypeName() string       { return ly.Type.String() }
func (ly *Layer) TypeNumber() int        { return int(ly.Type) }
func (ly *Layer) StyleObject() any       { return ly }
func (ly *Layer) NumRecvPaths() int      { return len(ly.RecvPaths) }
func (ly *Layer) NumSendPaths() int      { return len(ly.SendPaths) }
func (ly *Layer) UnitVarNum() int        { return NeuronVarN }
func (ly *Layer) UnitVarNames() []string { return NeuronVars }
func (ly *Layer) UnitVarProps() map[string]string {
	return NeuronVarProps
}

// RecvPath returns the receiving pathway at given index.
func (ly *Layer) RecvPath(idx int) emer.Path {
	return ly.RecvPaths[idx]
}

// SendPath returns the sending pathway at given index.
func (ly *Layer) SendPath(idx int) emer.Path {
	return ly.SendPaths[idx]
}

// UnitVarIndex returns the index of given variable name within
// the unit variable list, or -1 and error message if not found.
func (ly *Layer) UnitVarIndex(varNm string) (int, error) {
	return NeuronVarIndexByName(varNm)
}

// VarIndex1D returns the 1D index into the [NeuronVars] buffer
// for given unit and data-parallel indexes.
func (ly *Layer) VarIndex1D(ui, di int) int {
	return (di*ly.Shape.Len() + ui) * NeuronVarN
}

// UnitValue1D returns the value of the given variable index on the
// given unit, using a 1-dimensional unit index, and a data-parallel
// index di. Returns NaN on invalid index.
func (ly *Layer) UnitValue1D(varIndex int, idx, di int) float32 {
	nu := ly.Shape.Len()
	nv := len(ly.NeuronVars)
	if nv == 0 || nu == 0 {
		return math32.NaN()
	}
	nData := nv / (nu * NeuronVarN)
	if di < 0 || di >= nData || idx < 0 || idx >= nu {
		return math32.NaN()
	}
	if varIndex < 0 || varIndex >= NeuronVarN {
		return math32.NaN()
	}
	return ly.NeuronVars[(di*nu+idx)*NeuronVarN+varIndex]
}

// VarRange returns the min / max values for the given variable
// across all units and data-parallel values.
// Returns an error when the variable has no values (all NaN),
// e.g., because it is not supported by this layer type.
func (ly *Layer) VarRange(varNm string) (min, max float32, err error) {
	vi, err := NeuronVarIndexByName(varNm)
	if err != nil {
		return 0, 0, errors.Log(err)
	}
	nu := ly.Shape.Len()
	nData := len(ly.NeuronVars) / (nu * NeuronVarN)
	any := false
	for di := 0; di < nData; di++ {
		for ui := 0; ui < nu; ui++ {
			vl := ly.UnitValue1D(vi, ui, di)
			if math32.IsNaN(vl) {
				continue
			}
			if !any {
				min, max = vl, vl
				any = true
				continue
			}
			if vl < min {
				min = vl
			}
			if vl > max {
				max = vl
			}
		}
	}
	if !any {
		return 0, 0, fmt.Errorf("Layer %s: no values for variable: %s", ly.Name, varNm)
	}
	return
}

// ParamsString returns a listing of all parameters in the Layer and
// pathways within the layer. If nonDefault is true, only report those
// not at their default values.
func (ly *Layer) ParamsString(nonDefault bool) string {
	var b strings.Builder
	b.WriteString("////////  Layer: " + ly.Name + "\n")
	b.WriteString(ly.Params.ParamsString(nonDefault))
	for _, pt := range ly.RecvPaths {
		b.WriteString(pt.ParamsString(nonDefault))
	}
	return b.String()
}

// Defaults sets the default values for the Layer parameters.
func (ly *Layer) Defaults() {
	ly.Params.Layer = ly
	ly.Params.Defaults()
	for _, pt := range ly.RecvPaths {
		pt.Defaults()
	}
}

// UpdateParams updates all parameters given any changes that might have
// been made to individual values, and rebuilds the network model
// if any graph-affecting parameters (activation function, bias) changed.
func (ly *Layer) UpdateParams() {
	ly.Params.Update()
	for _, pt := range ly.RecvPaths {
		pt.UpdateParams()
	}
	if ly.Network != nil {
		ly.Network.UpdateParams()
	}
}

// varIndexDef returns the variable index of a unit variable
// from its name, with a default to 0 on error.
func varIndexDef(varNm string) int {
	vi, err := NeuronVarIndexByName(varNm)
	if err != nil {
		return 0
	}
	return vi
}

// hasRecvPaths returns true if this layer has any receiving paths,
// in which case it has a dense transformation and Bias values.
func (ly *Layer) hasRecvPaths() bool { return len(ly.RecvPaths) > 0 }

// setUnitVar sets a unit variable value in the [NeuronVars] buffer
// for given variable name, unit and data-parallel index.
func (ly *Layer) setUnitVar(varNm string, val float32, ui, di int) {
	nu := ly.Shape.Len()
	if ui < 0 || ui >= nu {
		return
	}
	vi, err := NeuronVarIndexByName(varNm)
	if err != nil {
		return
	}
	off := (di*nu+ui)*NeuronVarN + vi
	if off >= len(ly.NeuronVars) {
		return
	}
	ly.NeuronVars[off] = val
}

// AllocNeuronVars allocates the [NeuronVars] buffer to hold
// the given number of data-parallel values, if not already big enough.
func (ly *Layer) AllocNeuronVars(nData int) {
	nu := ly.Shape.Len()
	nt := nData * nu * NeuronVarN
	if cap(ly.NeuronVars) < nt {
		ly.NeuronVars = make([]float32, nt)
	} else {
		ly.NeuronVars = ly.NeuronVars[:nt]
	}
}

// InitWeights initializes the weight values for all paths
// receiving into this layer, using their [WtInitParams] settings.
func (ly *Layer) InitWeights() {
	for _, pt := range ly.RecvPaths {
		if pt.Off {
			continue
		}
		pt.InitWeights()
	}
}

// InitActs initializes all activation state values to zero.
func (ly *Layer) InitActs() {
	for i := range ly.NeuronVars {
		ly.NeuronVars[i] = 0
	}
}

// InitExt initializes the external input and target values to NaN,
// indicating that they have not yet been applied.
func (ly *Layer) InitExt() {
	nu := ly.Shape.Len()
	nData := len(ly.NeuronVars) / (nu * NeuronVarN)
	nan := math32.NaN()
	for di := 0; di < nData; di++ {
		for ui := 0; ui < nu; ui++ {
			ly.setUnitVar("Ext", nan, ui, di)
			ly.setUnitVar("Targ", nan, ui, di)
		}
	}
}

// ApplyExt applies an external input pattern of values, given as a
// [tensor.Tensor] (typically from a [table.Table] via the [env] system),
// for given data-parallel index di.
// For input layers, this sets the Ext, Act and Net values;
// for target layers, it sets the Targ values.
// It is a no-op for other layer types.
//
//	table = github.com/emer/emergent/v2/table
//	env = github.com/emer/emergent/v2/env
func (ly *Layer) ApplyExt(di int, st tensor.Tensor) {
	nu := ly.Shape.Len()
	if di < 0 || st.Len() < nu {
		return
	}
	switch ly.Type {
	case Input:
		for ui := range nu {
			vl := float32(st.Float1D(ui))
			ly.setUnitVar("Ext", vl, ui, di)
			ly.setUnitVar("Act", vl, ui, di)
			ly.setUnitVar("Net", vl, ui, di)
		}
	case Target:
		for ui := range nu {
			vl := float32(st.Float1D(ui))
			ly.setUnitVar("Targ", vl, ui, di)
		}
	}
}

// applyInputValues copies the current Ext values into the Act and Net
// values for an input layer, for given data-parallel index di.
// This is called by the network [ApplyExts] at the start of a trial.
func (ly *Layer) applyInputValues(di int) {
	nu := ly.Shape.Len()
	off0 := di * nu * NeuronVarN
	viE := varIndexDef("Ext")
	for ui := range nu {
		vl := ly.NeuronVars[off0+ui*NeuronVarN+viE]
		ly.NeuronVars[off0+ui*NeuronVarN+varIndexDef("Act")] = vl
		ly.NeuronVars[off0+ui*NeuronVarN+varIndexDef("Net")] = vl
	}
}

// zeroActValues zeros the Act and Net values for given data-parallel index di.
// This is called by the network [ApplyExts] for non-input layers
// at the start of a trial.
func (ly *Layer) zeroActValues(di int) {
	nu := ly.Shape.Len()
	viA := varIndexDef("Act")
	viN := varIndexDef("Net")
	for ui := range nu {
		off := (di*nu + ui) * NeuronVarN
		if off >= len(ly.NeuronVars) {
			return
		}
		ly.NeuronVars[off+viA] = 0
		ly.NeuronVars[off+viN] = 0
	}
}

// readbackAct reads back the Act (and Net) values from the given
// output tensors, for all data-parallel values.
// The outputs are expected as 2D [NData, NUnits] tensors, in the
// same order as the layer's output nodes were registered:
// output 0 is the Net values, output 1 is the Act values.
func (ly *Layer) readbackAct(outs []*tensors.Tensor) {
	nu := ly.Shape.Len()
	nData := len(ly.NeuronVars) / (nu * NeuronVarN)
	for oi, tsr := range outs {
		if tsr.Shape().Rank() != 2 {
			continue
		}
		flat := tensors.MustCopyFlatData[float32](tsr)
		vi := varIndexDef(layerOutVarName(oi))
		nOut := tsr.Shape().Dim(1)
		for di := 0; di < nData && di < tsr.Shape().Dim(0); di++ {
			for ui := 0; ui < nu && ui < nOut; ui++ {
				off := (di*nu+ui)*NeuronVarN + vi
				if off >= len(ly.NeuronVars) {
					return
				}
				ly.NeuronVars[off] = flat[di*nOut+ui]
			}
		}
	}
}

// layerOutVarName returns the unit variable name for the given
// output node index of a non-input layer.
func layerOutVarName(oi int) string {
	if oi == 0 {
		return "Net"
	}
	return "Act"
}

// readbackBias reads back the bias values from the first path's
// bias variable into the Bias unit variable buffer, for all
// data-parallel values.
func (ly *Layer) readbackBias() {
	if !ly.hasRecvPaths() {
		return
	}
	pt := ly.RecvPaths[0]
	if pt.Bias == nil || !pt.Bias.HasValue() {
		return
	}
	nu := ly.Shape.Len()
	nData := len(ly.NeuronVars) / (nu * NeuronVarN)
	bias := tensors.MustCopyFlatData[float32](pt.Bias.MustValue())
	if len(bias) < nu {
		return
	}
	vi := varIndexDef("Bias")
	for di := 0; di < nData; di++ {
		for ui := 0; ui < nu; ui++ {
			off := (di*nu+ui)*NeuronVarN + vi
			if off >= len(ly.NeuronVars) {
				return
			}
			ly.NeuronVars[off] = bias[ui]
		}
	}
}

// buildLayer builds the [gomlx] computation graph portion for this layer,
// creating the weight variables and the Net / Act output nodes.
// It is called from the [Network] build process, and requires that
// the Act nodes of all sending layers have already been built.
// scope is the root model scope under which the path variables go.
func (ly *Layer) buildLayer(scope *Scope, inputs []*graph.Node) error {
	g := ly.inputGraph()
	if g == nil {
		return fmt.Errorf("Layer %s: cannot build -- no input graph found", ly.Name)
	}
	var net *graph.Node
	for _, pt := range ly.RecvPaths {
		ln, err := pt.buildDense(scope, g)
		if err != nil {
			return err
		}
		if net == nil {
			net = ln
			continue
		}
		net = graph.Add(net, ln)
	}
	if net == nil {
		return fmt.Errorf("Layer %s: no receiving paths to build", ly.Name)
	}
	ly.Net = net
	ly.Act = applyAct(ly.Params.Act, net)
	return nil
}

// inputGraph returns the graph to use for building, based on the
// graph of the first input or sending layer node available.
func (ly *Layer) inputGraph() *graph.Graph {
	if ly.Input != nil {
		return ly.Input.Graph()
	}
	for _, pt := range ly.RecvPaths {
		if pt.Send != nil && pt.Send.Act != nil {
			return pt.Send.Act.Graph()
		}
	}
	return nil
}

// applyAct applies the layer's activation function to the net input node,
// according to the [ActParams] settings.
func applyAct(ap ActParams, net *graph.Node) *graph.Node {
	if ap.Fn == LeakyReLU {
		return activation.LeakyReluWith(net, ap.LeakySlope)
	}
	return activation.Apply(ap.Fn.MLX(), net)
}

// RecvPathValues fills in the values of the given synapse variable name,
// for the pathway from the given sending layer and sending neuron 1D index,
// for all receiving neurons in this layer, into the given float32 slice
// (only resized if not big enough).
// If the given sending layer or index is invalid, all values are set to NaN.
func (ly *Layer) RecvPathValues(vals *[]float32, varNm string, sendLay emer.Layer, sendIndex1D int, pathType string) error {
	nn := ly.Shape.Len()
	if *vals == nil || cap(*vals) < nn {
		*vals = make([]float32, nn)
	} else if len(*vals) < nn {
		*vals = (*vals)[:nn]
	}
	nan := math32.NaN()
	for i := range *vals {
		(*vals)[i] = nan
	}
	if sendLay == nil {
		return fmt.Errorf("Layer %s: RecvPathValues: sending layer is nil", ly.Name)
	}
	var pt *Path
	if pathType != "" {
		elp, err := ly.RecvPathBySendNameType(sendLay.Label(), pathType)
		if err != nil {
			return errors.Log(err)
		}
		pt = elp.(*Path)
	} else {
		elp, err := ly.RecvPathBySendName(sendLay.Label())
		if err != nil {
			return errors.Log(err)
		}
		pt = elp.(*Path)
	}
	if sendIndex1D < 0 || sendIndex1D >= pt.Send.Shape.Len() {
		return fmt.Errorf("Layer %s: RecvPathValues: invalid sending index: %d", ly.Name, sendIndex1D)
	}
	vi, err := pt.SynVarIndex(varNm)
	if err != nil {
		return errors.Log(err)
	}
	nRecv := ly.Shape.Len()
	for ri := range *vals {
		(*vals)[ri] = pt.SynValue1D(vi, sendIndex1D*nRecv+ri)
	}
	return nil
}

// SendPathValues fills in the values of the given synapse variable name,
// for the pathway into the given receiving layer and receiving neuron 1D index,
// for all sending neurons in this layer, into the given float32 slice
// (only resized if not big enough).
// If the given receiving layer or index is invalid, all values are set to NaN.
func (ly *Layer) SendPathValues(vals *[]float32, varNm string, recvLay emer.Layer, recvIndex1D int, pathType string) error {
	nn := ly.Shape.Len()
	if *vals == nil || cap(*vals) < nn {
		*vals = make([]float32, nn)
	} else if len(*vals) < nn {
		*vals = (*vals)[:nn]
	}
	nan := math32.NaN()
	for i := range *vals {
		(*vals)[i] = nan
	}
	if recvLay == nil {
		return fmt.Errorf("Layer %s: SendPathValues: receiving layer is nil", ly.Name)
	}
	var pt *Path
	if pathType != "" {
		elp, err := ly.SendPathByRecvNameType(recvLay.Label(), pathType)
		if err != nil {
			return errors.Log(err)
		}
		pt = elp.(*Path)
	} else {
		elp, err := ly.SendPathByRecvName(recvLay.Label())
		if err != nil {
			return errors.Log(err)
		}
		pt = elp.(*Path)
	}
	if recvIndex1D < 0 || recvIndex1D >= pt.Recv.Shape.Len() {
		return fmt.Errorf("Layer %s: SendPathValues: invalid receiving index: %d", ly.Name, recvIndex1D)
	}
	vi, err := pt.SynVarIndex(varNm)
	if err != nil {
		return errors.Log(err)
	}
	nRecv := pt.Recv.Shape.Len()
	for si := range *vals {
		(*vals)[si] = pt.SynValue1D(vi, si*nRecv+recvIndex1D)
	}
	return nil
}

///////////////////////////////////////////////////////////////////////
//  Weights File

// WriteWeightsJSON writes the weights from this layer from the
// receiver-side perspective in a JSON text format.
// Unit-level Bias values are saved as well, along with all
// the receiving path weights.
func (ly *Layer) WriteWeightsJSON(w io.Writer, depth int) {
	ly.WriteWeightsJSONBase(w, depth, "Bias")
}

// SetWeights sets the weights for this layer from the given
// [weights.Layer] decoded values, setting the bias variable
// values and all the receiving path weights.
func (ly *Layer) SetWeights(lw *weights.Layer) error {
	if ly.Off {
		return nil
	}
	if lw.Units != nil {
		if bv, ok := lw.Units["Bias"]; ok {
			ly.setBiasValues(bv)
		}
	}
	var err error
	rpts := ly.RecvPaths
	if len(lw.Paths) == len(rpts) { // this is essential if multiple paths from same layer
		for pi := range lw.Paths {
			pw := &lw.Paths[pi]
			pt := (rpts)[pi]
			er := pt.SetWeights(pw)
			if er != nil {
				err = er
			}
		}
	} else {
		for pi := range lw.Paths {
			pw := &lw.Paths[pi]
			elp, er := ly.RecvPathBySendName(pw.From)
			if er != nil {
				err = er
				continue
			}
			er = elp.(*Path).SetWeights(pw)
			if er != nil {
				err = er
			}
		}
	}
	return err
}

// setBiasValues sets the bias variable values from a flat list
// of values for each receiving unit, and updates the unit buffer.
func (ly *Layer) setBiasValues(vals []float32) {
	nu := ly.Shape.Len()
	if len(vals) < nu || !ly.hasRecvPaths() {
		return
	}
	pt := ly.RecvPaths[0]
	if pt.Bias == nil {
		return
	}
	err := pt.Bias.SetValue(tensors.FromFlatDataAndDimensions(vals, nu))
	if err != nil {
		errors.Log(err)
		return
	}
	ly.readbackBias()
}
