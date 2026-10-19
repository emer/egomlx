// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package egomlx

import (
	"fmt"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/math32"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/paths"
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/train/optimizer"

	_ "github.com/gomlx/gomlx/backends/default" // registers the pure-Go CPU backend
)

// DWtTypes controls how much of the weight-change (DWt) information
// is read back to the host during a [Network] [Network.Backward] pass.
// Computing the gradients is required for learning in any case, but
// reading back the values to the CPU for display and inspection has
// additional cost, so it should only be requested when actually needed.
type DWtTypes int32 //enums:enum

const (
	// DWtNone does not read back any DWt values -- only the overall
	// loss value is returned: the most efficient option.
	DWtNone DWtTypes = iota

	// DWtReadRaw reads back the raw gradients (pre-optimizer) computed
	// for each learning pathway, into the [Path] [Path.KernelDWt] and
	// [Path.BiasDWt] buffers, along with the updated weight values.
	DWtReadRaw
)

// gradOpt is the [gomlx] optimizer interface for applying updates from
// pre-computed gradient nodes, which the [gomlx] SGD and Adam optimizers
// both implement. It lets the backward pass return the gradients as graph
// outputs while still using the standard [gomlx] optimizers for updates.
//
//	optimizer = github.com/gomlx/gomlx/ml/train/optimizer
type gradOpt interface {
	UpdateGraphWithGradients(scope *model.Scope, gradients []*graph.Node, lossDType dtypes.DType)
}

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

	// Params has the global Network-level parameters, including the
	// loss function and optimizer settings for the backward pass.
	Params NetworkParams `display:"inline"`

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

	// TrainExec is the [gomlx] model executor for the training graph:
	// forward pass + loss + backward gradients + optimizer weight
	// updates, all in one compiled computation graph.
	// Created by [Build] if the network has any Target layers.
	TrainExec *model.Exec `display:"-"`

	// TrainVisExec is like [TrainExec], but it also returns the raw
	// DWt gradient values as graph outputs, for readback into the
	// [Path] buffers for visualization, which has added cost and so
	// is only run for [Backward] with [DWtReadRaw].
	TrainVisExec *model.Exec `display:"-"`

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

	// TargetLayers are the [Target] type layers, in order, among the
	// OutputLayers -- their Targ values are additional inputs to the
	// training graph, driving the loss for the backward pass.
	TargetLayers []*Layer `display:"-"`

	// Loss is the value of the loss function from the most recent
	// [Backward] pass, averaged over the current [NData] items.
	Loss float32 `display:"-"`

	// gradPaths records the pathway (and bias) correspondence for each
	// DWt gradient output of the training graph, in output order,
	// established when the training graph is built, so the gradient
	// readback can assign them to the right Path buffers.
	gradPaths []gradPath `display:"-"`

	// inBuf is a scratch buffer for the current input values,
	// of size MaxData * InputSize, for building the input tensors.
	inBuf []float32 `display:"-"`

	// targBuf is a scratch buffer for the current target values,
	// of size MaxData * TargSize, for building the target tensors.
	targBuf []float32 `display:"-"`
}

// gradPath records which pathway variable a DWt gradient
// graph output corresponds to.
type gradPath struct {
	// Pt is the pathway whose weights produced the gradient.
	Pt *Path

	// IsBias indicates whether the gradient is for the Bias
	// (true) or the Kernel (false) variable.
	IsBias bool
}

// NewNetwork returns a new egomlx Network with the given name,
// using the first registered [gomlx] compute backend
// (the pure-Go CPU backend by default).
func NewNetwork(name string) *Network {
	nt := &Network{}
	emer.InitNetwork(nt, name)
	nt.Backend = compute.MustNew()
	nt.Store = model.NewStore()
	nt.Params.Network = nt
	nt.Params.Defaults()
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
			ln := "NoLearn"
			if pt.Params.Learn {
				ln = "Learn"
			}
			s += pt.Name + ": " + pt.Params.Type.String() + " " + ub + " " + ln + " WtInit:" + pt.Params.WtInit.Init.String()
		}
	}
	return s
}

// KeyNetworkParams returns a listing of the most important
// network-level parameters, controlling the overall
// loss function and optimizer for learning.
func (nt *Network) KeyNetworkParams() string {
	return fmt.Sprintf("Loss: %s  Optim: %s  LRate: %g", nt.Params.Loss.String(), nt.Params.Optim.String(), nt.Params.LRate)
}

// Defaults sets default parameter values for everything in the Network.
func (nt *Network) Defaults() {
	nt.Params.Network = nt
	nt.Params.Defaults()
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
// [model] [model.Exec] executor for running forward passes,
// along with the training graph executors if the network has any
// [Target] layers.
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
	nt.Params.Network = nt
	nt.Params.SetStoreParams(nt.Store)
	nt.TargetLayers = nil
	for _, ly := range nt.OutputLayers {
		if ly.Type == Target {
			nt.TargetLayers = append(nt.TargetLayers, ly)
		}
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
	err = nt.buildTrainGraph()
	if err != nil {
		return err
	}
	return nt.preCompile()
}

// preCompile builds and compiles the computation graphs for the
// current [MaxData] shapes, so that the weight variables exist
// immediately after [Build], and can be initialized with [InitWeights]
// before the first [Forward] or [Backward] call.
// Graphs for other [NData] shapes are compiled lazily on demand.
func (nt *Network) preCompile() error {
	if nt.MaxData < 1 || len(nt.InputLayers) == 0 {
		return nil
	}
	inShapes := make([]shapes.Shape, 0, len(nt.InputLayers))
	for _, ly := range nt.InputLayers {
		inShapes = append(inShapes, shapes.Make(dtypes.Float32, nt.MaxData, ly.Shape.Len()))
	}
	if nt.Exec != nil {
		if _, err := nt.Exec.Compile(inShapes...); err != nil {
			return fmt.Errorf("Network %s: preCompile: %w", nt.Name, err)
		}
	}
	if nt.TrainExec == nil && nt.TrainVisExec == nil {
		return nil
	}
	trainShapes := make([]shapes.Shape, 0, len(inShapes)+len(nt.TargetLayers))
	trainShapes = append(trainShapes, inShapes...)
	for _, ly := range nt.TargetLayers {
		trainShapes = append(trainShapes, shapes.Make(dtypes.Float32, nt.MaxData, ly.Shape.Len()))
	}
	if nt.TrainExec != nil {
		if _, err := nt.TrainExec.Compile(trainShapes...); err != nil {
			return fmt.Errorf("Network %s: preCompile: train: %w", nt.Name, err)
		}
	}
	if nt.TrainVisExec != nil {
		if _, err := nt.TrainVisExec.Compile(trainShapes...); err != nil {
			return fmt.Errorf("Network %s: preCompile: train-vis: %w", nt.Name, err)
		}
	}
	return nil
}

// buildTrainGraph builds the training graph executors: one that runs
// forward + loss + backward + optimizer updates and returns only the
// loss, and one that additionally returns the raw DWt gradient values,
// for readback and visualization. These are only built if the network
// has [Target] layers to drive the loss. The graph itself is compiled
// on the first call to [Backward], for the given NData shapes.
func (nt *Network) buildTrainGraph() error {
	nt.TrainExec = nil
	nt.TrainVisExec = nil
	nt.gradPaths = nil
	if len(nt.TargetLayers) == 0 {
		return nil
	}
	mkExec := func(vis bool) (*model.Exec, error) {
		fn := func(scope *model.Scope, g *graph.Graph, inputs []*graph.Node) []*graph.Node {
			return nt.buildTrainFn(scope, g, inputs, vis)
		}
		return model.NewExec(nt.Backend, nt.Store, fn)
	}
	ex, err := mkExec(false)
	if err != nil {
		return fmt.Errorf("Network %s: buildTrainGraph: %w", nt.Name, err)
	}
	nt.TrainExec = ex
	vex, err := mkExec(true)
	if err != nil {
		return fmt.Errorf("Network %s: buildTrainGraph: vis: %w", nt.Name, err)
	}
	nt.TrainVisExec = vex
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

// gatherUnitVar gathers the values of the given unit variable name
// from the layer, for the current [NData] data items, into a flat
// buffer of size nData * NUnits, using a copy of the data.
// It uses the given scratch buffer, growing it as needed.
func (nt *Network) gatherUnitVar(ly *Layer, varNm string, buf []float32) ([]float32, error) {
	nData := nt.NData
	nu := ly.Shape.Len()
	if cap(buf) < nData*nu {
		buf = make([]float32, nData*nu)
	} else {
		buf = buf[:nData*nu]
	}
	vi := varIndexDef(varNm)
	for di := 0; di < nData; di++ {
		for ui := 0; ui < nu; ui++ {
			off := (di*nu+ui)*NeuronVarN + vi
			if off >= len(ly.NeuronVars) {
				return nil, fmt.Errorf("Network: layer %s: NeuronVars buffer too small for NData = %d", ly.Name, nData)
			}
			buf[di*nu+ui] = ly.NeuronVars[off]
		}
	}
	return buf, nil
}

// inputArgs builds the input tensors for the current [NData] data
// items, from the Act values of the [InputLayers].
func (nt *Network) inputArgs() ([]any, error) {
	var args []any
	var err error
	for _, ly := range nt.InputLayers {
		nt.inBuf, err = nt.gatherUnitVar(ly, "Act", nt.inBuf)
		if err != nil {
			return nil, err
		}
		nu := ly.Shape.Len()
		args = append(args, tensors.FromFlatDataAndDimensions(nt.inBuf, nt.NData, nu))
	}
	return args, nil
}

// targArgs builds the target tensors for the current [NData] data
// items, from the Targ values of the [TargetLayers].
// It returns an error if any target values are unset (NaN), which
// indicates that [ApplyExts] has not been called with target inputs
// for the current trial.
func (nt *Network) targArgs() ([]any, error) {
	var args []any
	var err error
	for _, ly := range nt.TargetLayers {
		nt.targBuf, err = nt.gatherUnitVar(ly, "Targ", nt.targBuf)
		if err != nil {
			return nil, err
		}
		for _, v := range nt.targBuf {
			if math32.IsNaN(v) {
				return nil, fmt.Errorf("target layer %s: Targ values not applied -- call ApplyExt with target values before Backward", ly.Name)
			}
		}
		nu := ly.Shape.Len()
		args = append(args, tensors.FromFlatDataAndDimensions(nt.targBuf, nt.NData, nu))
	}
	return args, nil
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
	args, err := nt.inputArgs()
	if err != nil {
		return errors.Log(fmt.Errorf("Network: Forward: %w", err))
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

// targetReachable returns a map of layers from which a sending
// pathway chain reaches any [Target] layer, and therefore contribute
// to the loss driving the backward pass. Layers whose outputs are not
// used by any Target layer have no gradient, and their weights are
// marked non-trainable during training.
func (nt *Network) targetReachable() map[*Layer]bool {
	reach := make(map[*Layer]bool, len(nt.Layers))
	for _, ly := range nt.TargetLayers {
		reach[ly] = true
	}
	for i := len(nt.Layers) - 1; i >= 0; i-- {
		ly := nt.Layers[i]
		if !reach[ly] {
			continue
		}
		for _, pt := range ly.RecvPaths {
			reach[pt.Send] = true
		}
	}
	return reach
}

// buildLoss builds the [gomlx] computation graph for the scalar loss,
// computed between the Act nodes of the [TargetLayers] and the
// corresponding target input nodes, according to the [NetworkParams.Loss]
// function, which must use the same inputs + targets argument ordering.
func (nt *Network) buildLoss(g *graph.Graph, targs []*graph.Node) *graph.Node {
	var total *graph.Node
	nData := 0
	for ti, ly := range nt.TargetLayers {
		if ly.Act == nil {
			panic(fmt.Sprintf("buildLoss: target layer %s has no activation node -- build it first", ly.Name))
		}
		targ := targs[ti]
		if nData == 0 {
			nData = targ.Shape().Dim(0)
		}
		err := graph.Sub(ly.Act, targ)
		var l *graph.Node
		switch nt.Params.Loss {
		case SSE:
			l = graph.ReduceAllSum(graph.Square(err))
		case MSE:
			l = graph.ReduceAllMean(graph.Square(err))
		case BCE:
			// clip the predictions into (0, 1) for log stability:
			a := graph.ClipScalar(ly.Act, 1e-7, 1-1e-7)
			ce := graph.Neg(graph.Add(
				graph.Mul(targ, graph.Log(a)),
				graph.Mul(graph.OneMinus(targ), graph.Log(graph.OneMinus(a)))))
			l = graph.ReduceAllMean(ce)
		default:
			l = graph.ReduceAllSum(graph.Square(err))
		}
		if total == nil {
			total = l
			continue
		}
		total = graph.Add(total, l)
	}
	if nt.Params.Loss == SSE && nData > 0 {
		total = graph.MulScalar(total, 0.5/float64(nData))
	}
	return total
}

// buildTrainFn builds the computation graph for one training pass:
// the forward model, the loss, the gradients with respect to all
// trainable pathway variables, and the optimizer weight updates --
// all in one graph. It is called by the training [model] [model.Exec]
// mechanism, potentially for each different input shape (NData).
//
// Weight variables of pathways that are learning ([PathParams.Learn])
// and lie on a chain feeding a [Target] layer are marked trainable,
// so the [gomlx] optimizer updates exactly those weights. The updated
// variable values are written back to the [Store] automatically by the
// [model.Exec] machinery. The graph returns the scalar loss, and if vis
// is true, additionally the raw DWt gradient nodes, in the order
// recorded in [Network.gradPaths], for readback and visualization.
func (nt *Network) buildTrainFn(scope *model.Scope, g *graph.Graph, inputs []*graph.Node, vis bool) []*graph.Node {
	nIn := len(nt.InputLayers)
	nTarg := len(nt.TargetLayers)
	if nIn+nTarg > len(inputs) {
		panic(fmt.Sprintf("buildTrainFn: expected %d inputs (%d layers + %d targets), got %d", nIn+nTarg, nIn, nTarg, len(inputs)))
	}
	nt.buildModelFn(scope, g, inputs[:nIn])
	loss := nt.buildLoss(g, inputs[nIn:])

	opt := nt.Params.Optimizer()
	reach := nt.targetReachable()
	nt.gradPaths = nil
	var varNodes []*graph.Node
	for _, ly := range nt.Layers {
		for _, pt := range ly.RecvPaths {
			train := pt.learnable() && reach[pt.Recv]
			if pt.Kernel != nil {
				pt.Kernel.SetTrainable(train)
				if train && pt.Kernel.InUseByGraph(g) {
					varNodes = append(varNodes, pt.Kernel.NodeValue(g))
					nt.gradPaths = append(nt.gradPaths, gradPath{Pt: pt})
				}
			}
			if pt.Bias != nil {
				pt.Bias.SetTrainable(train && pt.Params.UseBias)
				if train && pt.Params.UseBias && pt.Bias.InUseByGraph(g) {
					varNodes = append(varNodes, pt.Bias.NodeValue(g))
					nt.gradPaths = append(nt.gradPaths, gradPath{Pt: pt, IsBias: true})
				}
			}
		}
	}
	if len(varNodes) == 0 {
		// nothing to learn: return just the loss.
		return []*graph.Node{loss}
	}
	grads := graph.Gradient(loss, varNodes...)
	if gopt, ok := opt.(gradOpt); ok {
		gopt.UpdateGraphWithGradients(scope, grads, loss.DType())
	} else {
		opt.UpdateGraph(scope, g, loss)
	}
	outs := []*graph.Node{loss}
	if vis {
		outs = append(outs, grads...)
	}
	return outs
}

// Backward runs one backward learning pass of the model on the current
// input data and target values (as set by [ApplyExts]), for the current
// [NData] data items. It computes the loss between the Act and Targ
// values of the [TargetLayers], backpropagates the error gradients
// through the model with [gomlx] automatic differentiation, and applies
// the weight updates using the optimizer configured in [NetworkParams],
// all in one compiled computation graph.
//
// The training graph recomputes the forward pass internally to obtain
// the gradients; the activation value buffers are not re-read (they
// hold the values from the last [Forward] pass, which are identical
// for unchanged inputs). The current [Network.Loss] value is read
// back, and set to NaN if the loss is not finite, which typically
// indicates a numerical instability.
//
// By default ([DWtNone]) the DWt gradient values are not read back to
// the host, which is the efficient option for training. Pass
// [DWtReadRaw] to read the raw gradients into the [Path] [Path.KernelDWt]
// and [Path.BiasDWt] buffers, along with the updated weight values,
// for visualization and inspection (this has added cost, so request it
// only when actually needed, e.g., when the [netview] is showing
// synapse values).
//
// [netview] = github.com/emer/emergent/v2/netview
func (nt *Network) Backward(updtDWt ...DWtTypes) error {
	dwts := DWtNone
	if len(updtDWt) > 0 {
		dwts = updtDWt[0]
	}
	if nt.Exec == nil {
		return errors.New("Network: not built -- call Build first")
	}
	if nt.TrainExec == nil {
		return fmt.Errorf("Network: Backward: no %s layers to compute loss -- a learning network must have at least one Target layer", Target.String())
	}
	inArgs, err := nt.inputArgs()
	if err != nil {
		return errors.Log(fmt.Errorf("Network: Backward: %w", err))
	}
	tArgs, err := nt.targArgs()
	if err != nil {
		return errors.Log(fmt.Errorf("Network: Backward: %w", err))
	}
	args := append(inArgs, tArgs...)
	ex := nt.TrainExec
	if dwts == DWtReadRaw && nt.TrainVisExec != nil {
		ex = nt.TrainVisExec
	}
	outs, err := ex.Call(args...)
	if err != nil {
		return errors.Log(fmt.Errorf("Network: Backward: %w", err))
	}
	nt.Loss = readScalar(outs[0])
	if math32.IsNaN(nt.Loss) {
		errors.Log(fmt.Errorf("Network: Backward: loss is NaN -- check input / target values and network stability"))
	}
	if dwts == DWtReadRaw {
		for gi, gp := range nt.gradPaths {
			oi := gi + 1
			if oi >= len(outs) {
				break
			}
			gv := tensors.MustCopyFlatData[float32](outs[oi])
			if gp.IsBias {
				if len(gv) == len(gp.Pt.BiasDWt) {
					copy(gp.Pt.BiasDWt, gv)
				}
			} else if len(gv) == len(gp.Pt.KernelDWt) {
				copy(gp.Pt.KernelDWt, gv)
			}
		}
		// weights were changed by the optimizer in the graph:
		// read back the updated values for display and access.
		nt.ReadWeights()
	}
	return nil
}

// readScalar returns the float32 value of a scalar output tensor.
func readScalar(tsr *tensors.Tensor) float32 {
	if tsr.Shape().Rank() != 0 {
		return math32.NaN()
	}
	fl := tensors.MustCopyFlatData[float32](tsr)
	if len(fl) != 1 {
		return math32.NaN()
	}
	return fl[0]
}

// TrainTrial runs one full training trial: a forward pass, followed by
// a backward pass that computes the gradients and updates the weights
// with the optimizer, for the current input and target data.
// The optional updtDWt controls the DWt readback during Backward:
// see [Network.Backward].
func (nt *Network) TrainTrial(updtDWt ...DWtTypes) error {
	err := nt.Forward()
	if err != nil {
		return err
	}
	return nt.Backward(updtDWt...)
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

// GlobalStep returns the number of optimizer update steps taken so far,
// as tracked by the [gomlx] global step variable in the [Store],
// which is incremented once per [Backward] pass that updates weights.
// Returns 0 if there is no store.
func (nt *Network) GlobalStep() int64 {
	if nt.Store == nil {
		return 0
	}
	return optimizer.GetGlobalStep(nt.Store)
}
