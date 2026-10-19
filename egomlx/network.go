// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package egomlx

import (
	"fmt"

	"cogentcore.org/core/base/errors"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/paths"
	"github.com/gomlx/compute"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"

	_ "github.com/gomlx/gomlx/backends/default" // registers the pure-Go CPU backend
)

// Network implements the [emer.Network] interface for a
// [gomlx] machine learning model, providing the standard
// emergent functionality for visualization in the [netview],
// parameter setting, and weights I/O.
//
// The model is built using the standard [gomlx] API: layers
// are created and connected with [paths.Pattern] connectivity,
// and the [Build] method compiles the [gomlx] computation graph
// and a [model] [model.Exec] executor for running forward passes.
//
// # Data Parallelism
//
// The network supports processing multiple data items (input patterns)
// in parallel, batched along the outer-most dimension of all state,
// consistent with the axon convention. [MaxData] sets the maximum
// number of parallel data values that buffers are allocated for,
// and [NData] is the current number of items actually processed,
// up to MaxData.
type Network struct {
	emer.NetworkBase

	// Layers is the list of layers in the network.
	Layers []*Layer

	// LayerClassMap is a map from class name to layer names,
	// updated by [UpdateLayerMaps]. The layer Type is always
	// included as a Class, along with any other space-separated
	// strings in [LayerBase.Class].
	LayerClassMap map[string][]string `display:"-"`

	// Backend is the [gomlx] compute backend used for the model.
	// By default, this is the first registered backend, which is
	// the pure-Go CPU backend (gobackend), registered by the
	// gomlx backends/default package. Use [SetBackend] to select
	// a different backend, e.g., an XLA CPU or GPU backend.
	Backend compute.Backend `display:"-"`

	// Store is the [gomlx] variable store for the model,
	// holding all the weight and other variables.
	Store *model.Store `display:"-"`

	// Exec is the [gomlx] model executor, which runs the
	// compiled computation graph. Created by [Build].
	Exec *model.Exec `display:"-"`

	// MaxData is the maximum number of data items that can be
	// processed in parallel -- the size of the state buffers.
	MaxData int `edit:"-" min:"1"`

	// NData is the current number of data items being
	// processed in parallel, up to MaxData.
	NData int `min:"1"`

	// InputLayers are the input layers, in order -- the
	// graph inputs. Their Act values are the applied external
	// input patterns, with no transformation.
	InputLayers []*Layer `display:"-"`

	// OutputLayers are the non-input layers, in order --
	// the graph outputs, of the Net and Act values.
	OutputLayers []*Layer `display:"-"`

	// inBuf is a scratch buffer for the current input values,
	// of size MaxData * InputSize, for building the input tensors.
	inBuf []float32 `display:"-"`
}

// NewNetwork returns a new egomlx Network with the given name,
// using the first registered [gomlx] compute backend
// (the pure-Go CPU backend by default).
func NewNetwork(name string) *Network {
	nt := &Network{}
	emer.InitNetwork(nt, name)
	nt.Backend = compute.MustNew()
	nt.Store = model.NewStore()
	nt.SetMaxData(1)
	nt.RandSeed = 0
	nt.ResetRandSeed()
	return nt
}

// emer.Network interface methods

func (nt *Network) NumLayers() int { return len(nt.Layers) }
func (nt *Network) EmerLayer(idx int) emer.Layer {
	return nt.Layers[idx]
}
func (nt *Network) MaxParallelData() int { return nt.MaxData }
func (nt *Network) NParallelData() int   { return nt.NData }

// VarCategories returns a list of unit and synapse variable categories,
// which organizes the variables into separate tabs in the network view.
func (nt *Network) VarCategories() []emer.VarCategory {
	return []emer.VarCategory{
		{Cat: "Unit", Doc: "unit-level variables"},
		{Cat: "Wt", Doc: "synapse-level weight variables"},
	}
}

// UnitVarNames returns a list of variable names available on
// the units in this network.
func (nt *Network) UnitVarNames() []string {
	return NeuronVars
}

// UnitVarProps returns a map of unit variable properties.
func (nt *Network) UnitVarProps() map[string]string {
	return NeuronVarProps
}

// SynVarNames returns the names of all the variables on the
// synapses in this network.
func (nt *Network) SynVarNames() []string {
	return SynapseVars
}

// SynVarProps returns a map of synapse variable properties.
func (nt *Network) SynVarProps() map[string]string {
	return SynapseVarProps
}

// KeyLayerParams returns a listing for all layers in the network,
// of the most important layer-level parameters.
func (nt *Network) KeyLayerParams() string {
	var s string
	for _, ly := range nt.Layers {
		if s != "" {
			s += "\n"
		}
		s += ly.Name + ": " + ly.Type.String() + " " + ly.Params.Act.Fn.String()
	}
	return s
}

// KeyPathParams returns a listing for all pathways in the network,
// of the most important pathway-level parameters.
func (nt *Network) KeyPathParams() string {
	var s string
	for _, ly := range nt.Layers {
		for _, pt := range ly.RecvPaths {
			if s != "" {
				s += "\n"
			}
			ub := "NoBias"
			if pt.Params.UseBias {
				ub = "Bias"
			}
			s += pt.Name + ": " + pt.Params.Type.String() + " " + ub + " WtInit:" + pt.Params.WtInit.Init.String()
		}
	}
	return s
}

// Defaults sets default parameter values for everything in the Network.
func (nt *Network) Defaults() {
	for _, ly := range nt.Layers {
		ly.Defaults()
	}
}

// UpdateParams updates parameter values for all Network parameters,
// based on any other params that might have changed,
// rebuilding the model graph as needed.
func (nt *Network) UpdateParams() {
	nt.buildGraph()
}

// SetMaxData sets the [MaxData] and current [NData] to the given value,
// re-allocating the state buffers.
func (nt *Network) SetMaxData(maxData int) {
	if maxData < 1 {
		maxData = 1
	}
	nt.MaxData = maxData
	nt.SetNData(maxData)
}

// SetNData sets the current number of data items to process in parallel,
// up to [MaxData].
func (nt *Network) SetNData(nData int) {
	if nData < 1 {
		nData = 1
	}
	if nData > nt.MaxData {
		nData = nt.MaxData
	}
	nt.NData = nData
}

// SetBackend sets the [gomlx] compute backend to use,
// and rebuilds the model graph.
func (nt *Network) SetBackend(b compute.Backend) {
	nt.Backend = b
	nt.buildGraph()
}

// AddLayer2D adds a new 2D layer with the given name, type,
// and Y, X shape dimensions, returning it.
// The layer should then be connected with [ConnectLayers].
func (nt *Network) AddLayer2D(name string, tp LayerTypes, y, x int) *Layer {
	ly := &Layer{Type: tp, Network: nt}
	emer.InitLayer(ly, name)
	ly.SetShape(y, x)
	ly.Params.Layer = ly
	nt.Layers = append(nt.Layers, ly)
	nt.UpdateLayerNameMap()
	return ly
}

// AddLayer1D adds a new 1D layer with the given name, type,
// and number of units, returning it.
// The layer should then be connected with [ConnectLayers].
func (nt *Network) AddLayer1D(name string, tp LayerTypes, n int) *Layer {
	ly := &Layer{Type: tp, Network: nt}
	emer.InitLayer(ly, name)
	ly.SetShape(n)
	ly.Params.Layer = ly
	nt.Layers = append(nt.Layers, ly)
	nt.UpdateLayerNameMap()
	return ly
}

// ConnectLayers connects the given sending and receiving layers
// with a new [Path] of the given type, using the given connectivity
// pattern, returning the path.
// The current toolkit version supports only the [paths.Full]
// full connectivity pattern, because [gomlx] dense layers
// are fully connected.
func (nt *Network) ConnectLayers(send, recv *Layer, pat paths.Pattern, typ PathTypes) *Path {
	if pat.Name() != "Full" {
		errors.Log(fmt.Errorf("ConnectLayers: %s -> %s: only the Full connectivity pattern is supported in the current egomlx toolkit, got: %s", send.Name, recv.Name, pat.Name()))
	}
	pt := &Path{}
	emer.InitPath(pt)
	pt.Connect(send, recv, pat, typ)
	pt.Params.Path = pt
	send.SendPaths = append(send.SendPaths, pt)
	recv.RecvPaths = append(recv.RecvPaths, pt)
	return pt
}

// Build builds the network: validates the layer and path structure,
// allocates the state buffers, builds the [gomlx] computation graph
// and [model] [model.Exec] executor, and computes the layer layout.
// It must be called after all layers and connections have been added,
// and after [SetMaxData].
func (nt *Network) Build() error {
	if len(nt.Layers) == 0 {
		return errors.New("Network: no layers to build")
	}
	for li, ly := range nt.Layers {
		ly.Index = li
	}
	nt.InputLayers = nil
	nt.OutputLayers = nil
	for _, ly := range nt.Layers {
		if len(ly.RecvPaths) == 0 {
			nt.InputLayers = append(nt.InputLayers, ly)
			continue
		}
		nt.OutputLayers = append(nt.OutputLayers, ly)
		for _, pt := range ly.RecvPaths {
			if pt.Send.Index >= pt.Recv.Index {
				return errors.New(fmt.Sprintf("Network: path %s: sending layer %s must come before receiving layer %s in a feedforward network", pt.Name, pt.Send.Name, pt.Recv.Name))
			}
		}
	}
	if len(nt.InputLayers) == 0 {
		return errors.New("Network: no input layers -- every feedforward network must have at least one input layer")
	}
	nt.UpdateLayerNameMap()
	nt.allocBuffers()
	err := nt.buildGraph()
	if err != nil {
		return err
	}
	nt.LayoutLayers()
	return nil
}

// allocBuffers allocates the state buffers for all layers and paths
// for the current [MaxData].
func (nt *Network) allocBuffers() {
	for _, ly := range nt.Layers {
		ly.AllocNeuronVars(nt.MaxData)
	}
	for _, ly := range nt.Layers {
		for _, pt := range ly.RecvPaths {
			pt.allocBuffers()
		}
	}
}

// buildGraph builds the [gomlx] computation graph and creates the
// [model] [model.Exec] executor for running forward passes.
// This is called by [Build], and re-called by [UpdateParams]
// and [SetBackend] when the graph structure may have changed.
// Note: this is cheap for the small models that the toolkit is
// typically used for.
func (nt *Network) buildGraph() error {
	if len(nt.Layers) == 0 {
		return nil
	}
	if nt.Store == nil {
		nt.Store = model.NewStore()
	}
	if nt.Backend == nil {
		nt.Backend = compute.MustNew()
	}
	modelFn := func(scope *model.Scope, g *graph.Graph, inputs []*graph.Node) []*graph.Node {
		return nt.buildModelFn(scope, g, inputs)
	}
	ex, err := model.NewExec(nt.Backend, nt.Store, modelFn)
	if err != nil {
		nt.Exec = nil
		return fmt.Errorf("Network %s: buildGraph: %w", nt.Name, err)
	}
	nt.Exec = ex
	return nil
}

// buildModelFn builds the computation graph for the model,
// given the runtime input nodes.
// It is called by the [model] [model.Exec] mechanism,
// potentially for each different input shape (NData).
// Weight variables are created on first build, and reused
// (with their current values) for subsequent builds.
func (nt *Network) buildModelFn(scope *model.Scope, g *graph.Graph, inputs []*graph.Node) []*graph.Node {
	ii := 0
	var outs []*graph.Node
	for _, ly := range nt.Layers {
		if len(ly.RecvPaths) == 0 { // input layer
			if ii >= len(inputs) {
				panic(fmt.Sprintf("buildModelFn: no input for layer %s", ly.Name))
			}
			ly.Input = inputs[ii]
			ly.Net = inputs[ii]
			ly.Act = inputs[ii]
			ii++
			continue
		}
		err := ly.buildLayer(scope, inputs)
		if err != nil {
			panic(err)
		}
		outs = append(outs, ly.Net, ly.Act)
	}
	return outs
}

// InitWeights initializes the weight values for all paths in the network,
// using their [WtInitParams] settings, and the network random number
// generator (see [NetworkBase.Rand]), for reproducible initialization.
// It also reads back the weight values so they are available
// for display, and initializes the activation buffers to zero.
func (nt *Network) InitWeights() {
	for _, ly := range nt.Layers {
		ly.InitWeights()
		ly.InitActs()
	}
	for _, ly := range nt.Layers {
		for _, pt := range ly.RecvPaths {
			if !pt.Off {
				pt.InitWeights()
			}
		}
	}
	nt.ReadWeights()
}

// InitActs initializes all activation state values to zero,
// and all external input / target values to NaN.
func (nt *Network) InitActs() {
	for _, ly := range nt.Layers {
		ly.InitActs()
	}
}

// InitExt initializes the external input and target values to NaN
// on all layers, indicating that they have not yet been applied.
func (nt *Network) InitExt() {
	for _, ly := range nt.Layers {
		ly.InitExt()
	}
}

// ApplyExts applies the external input values to the network,
// starting a new trial: for input layers, the Act and Net values
// are set to the current Ext values; for all other layers,
// the Act and Net values are zeroed, in preparation for the
// forward pass.
// This should be called after all [ApplyExt] calls for the
// current trial have been made.
func (nt *Network) ApplyExts() {
	for di := range nt.NData {
		for _, ly := range nt.Layers {
			if len(ly.RecvPaths) == 0 {
				ly.applyInputValues(di)
				continue
			}
			ly.zeroActValues(di)
		}
	}
}

// Forward runs one forward pass of the model on the current
// input data (as set by [ApplyExts]), for the current [NData]
// data items, computing all the activation values.
// It reads back the activation values and all the weight values,
// so that all network state is available for display and access.
func (nt *Network) Forward() error {
	if nt.Exec == nil {
		return errors.New("Network: not built -- call Build first")
	}
	nData := nt.NData
	var args []any
	for _, ly := range nt.InputLayers {
		nu := ly.Shape.Len()
		if len(nt.inBuf) < nData*nu {
			nt.inBuf = make([]float32, nData*nu)
		}
		buf := nt.inBuf[:nData*nu]
		viA := varIndexDef("Act")
		for di := 0; di < nData; di++ {
			for ui := 0; ui < nu; ui++ {
				off := (di*nu+ui)*NeuronVarN + viA
				if off >= len(ly.NeuronVars) {
					return fmt.Errorf("Network: layer %s: NeuronVars buffer too small for NData = %d", ly.Name, nData)
				}
				buf[di*nu+ui] = ly.NeuronVars[off]
			}
		}
		args = append(args, tensors.FromFlatDataAndDimensions(buf, nData, nu))
	}
	outs, err := nt.Exec.Call(args...)
	if err != nil {
		return errors.Log(fmt.Errorf("Network: Forward: %w", err))
	}
	for li, ly := range nt.OutputLayers {
		ly.readbackAct(outs[2*li : 2*li+2])
		ly.readbackBias()
	}
	nt.ReadWeights()
	return nil
}

// Backward runs backpropagation of error signals through the model,
// computing weight gradients.
// This is a stub -- not yet implemented in the current toolkit.
func (nt *Network) Backward() error {
	// todo: backprop with [gomlx] gradient computation
	return nil
}

// UpdateWeights applies weight updates based on the computed gradients,
// using the path learning rate parameters.
// This is a stub -- not yet implemented in the current toolkit.
func (nt *Network) UpdateWeights() {
	// todo: weight updates with a [gomlx] optimizer
}

// TrainTrial runs one full training trial: a forward pass,
// a backward pass, and a weight update, for the current
// input data.
// Backpropagation and weight updates are currently stubbed,
// so this is equivalent to a forward pass.
func (nt *Network) TrainTrial() error {
	err := nt.Forward()
	if err != nil {
		return err
	}
	err = nt.Backward()
	if err != nil {
		return err
	}
	nt.UpdateWeights()
	return nil
}

// TestTrial runs one test trial: a forward pass only,
// with no learning.
func (nt *Network) TestTrial() error {
	return nt.Forward()
}

// ReadWeights reads back the current weight variable values
// into the path readback buffers, so they are available
// for display and access without re-running the model.
func (nt *Network) ReadWeights() {
	for _, ly := range nt.Layers {
		for _, pt := range ly.RecvPaths {
			pt.ReadbackWeights()
		}
		ly.readbackBias()
	}
}
