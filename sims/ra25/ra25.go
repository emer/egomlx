// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// ra25 runs a simple random-associator four-layer egomlx network
// that uses the standard supervised backprop learning paradigm to learn
// mappings between 25 random input / output patterns
// defined over 5x5 input / output layers (i.e., 25 units).
//
// The network is built with the [gomlx] API and expressed through the
// [egomlx] toolkit, which implements the emergent [emer] interfaces
// for visualization in the [netview]. The forward pass, backpropagation
// of error gradients, and optimizer weight updates are all real,
// so the network genuinely learns, and the activations, error,
// DWt and weight values shown in the netview are live.
package ra25

//go:generate core generate -add-types -add-funcs

import (
	"embed"
	"fmt"
	"io/fs"
	"math"
	"os"
	"reflect"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/base/metadata"
	"cogentcore.org/core/core"
	"cogentcore.org/core/enums"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/math32"
	"cogentcore.org/core/tree"
	"cogentcore.org/lab/base/mpi"
	"cogentcore.org/lab/base/randx"
	"cogentcore.org/lab/patterns"
	"cogentcore.org/lab/plot"
	"cogentcore.org/lab/stats/stats"
	"cogentcore.org/lab/table"
	"cogentcore.org/lab/tensor"
	"cogentcore.org/lab/tensorfs"
	"github.com/emer/egomlx/egomlx"
	"github.com/emer/emergent/v2/egui"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/looper"
	"github.com/emer/emergent/v2/paths"
)

//go:embed random_5x5_25.tsv
var embedfs embed.FS

// Modes are the looping modes (Stacks) for running and statistics.
type Modes int32 //enums:enum

const (
	// Train mode: run the training pass of the model.
	Train Modes = iota

	// Test mode: run the testing pass of the model.
	Test
)

// Levels are the looping levels for running and statistics.
type Levels int32 //enums:enum

const (
	// Cycle is the shortest time-scale: a single forward pass.
	Cycle Levels = iota

	// Trial is one input pattern (or batch of NData patterns) processed.
	Trial

	// Epoch is one complete pass through the full set of trials.
	Epoch

	// Run is one complete training run, with its own random seed.
	Run

	// Expt is the outermost experiment level.
	Expt
)

const (
	// Start initializes stats, in start arg of StatFuncs call.
	Start = true
	// Step is an iteration of stats, in start arg of StatFuncs call.
	Step = false
)

// see params.go for params, config.go for config

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {

	// simulation configuration parameters -- set by .toml config file and / or args
	Config *Config `new-window:"+"`

	// Net is the network: click to view / edit parameters for layers, paths, etc.
	Net *egomlx.Network `new-window:"+" display:"no-inline"`

	// Params manages network parameter setting.
	Params egomlx.Params `display:"inline"`

	// Loops are the control loops for running the sim, in different Modes
	// across stacks of Levels.
	Loops *looper.Stacks `new-window:"+" display:"no-inline"`

	// Envs provides mode-string based storage of environments.
	Envs env.Envs `new-window:"+" display:"no-inline"`

	// TrainUpdate has Train mode netview update parameters.
	TrainUpdate egomlx.NetViewUpdate `display:"inline"`

	// TestUpdate has Test mode netview update parameters.
	TestUpdate egomlx.NetViewUpdate `display:"inline"`

	// Root is the root tensorfs directory, where all stats and other misc sim data goes.
	Root *tensorfs.Node `display:"-"`

	// Stats has the stats directory within Root.
	Stats *tensorfs.Node `display:"-"`

	// Current has the current stats values within Stats.
	Current *tensorfs.Node `display:"-"`

	// StatFuncs are statistics functions called at given mode and level,
	// to perform all stats computations. phase = Start does init at start of given level,
	// and all intialization / configuration (called during Init too).
	StatFuncs []func(mode enums.Enum, level enums.Enum, start bool) `display:"-"`

	// GUI manages all the GUI elements
	GUI egui.GUI `display:"-"`

	// RandSeeds is a list of random seeds to use for each run.
	RandSeeds randx.Seeds `display:"-"`
}

// SetConfig sets the sim config.
func (ss *Sim) SetConfig(cfg *Config) { ss.Config = cfg }

// Body returns the [core.Body] used by the sim.
func (ss *Sim) Body() *core.Body { return ss.GUI.Body }

// ConfigSim initializes the config-dependent parts of the sim.
func (ss *Sim) ConfigSim() {
	ss.Root, _ = tensorfs.NewDir("Root")
	tensorfs.CurRoot = ss.Root
	ss.Net = egomlx.NewNetwork(ss.Config.Name)
	ss.Params.Config(NetworkParams, LayerParams, PathParams, ss.Config.Params.Sheet, ss.Config.Params.Tag, "github.com/emer/egomlx/sims/ra25", reflect.ValueOf(ss))
	ss.RandSeeds.Init(100) // max 100 runs
	ss.InitRandSeed(0)
	// ss.ConfigInputs()
	ss.OpenInputs()
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigLoops()
	ss.ConfigStats()
	if ss.Config.Params.SaveAll {
		ss.Config.Params.SaveAll = false
		ss.Net.SaveParamsSnapshot(&ss.Config, ss.Config.Params.Good)
		os.Exit(0)
	}
}

// ConfigEnv configures the train and test environments,
// using the embedded 25 random input / output patterns.
func (ss *Sim) ConfigEnv() {
	// Can be called multiple times -- don't re-create
	var trn, tst *env.FixedTable
	if len(ss.Envs) == 0 {
		trn = &env.FixedTable{}
		tst = &env.FixedTable{}
	} else {
		trn = ss.Envs.ByMode(Train).(*env.FixedTable)
		tst = ss.Envs.ByMode(Test).(*env.FixedTable)
	}

	inputs := tensorfs.DirTable(ss.Root.Dir("Inputs/Train"), nil)

	// note: names must be standard here!
	trn.Name = Train.String()
	trn.Config(table.NewView(inputs))
	trn.Validate()

	tst.Name = Test.String()
	tst.Config(table.NewView(inputs))
	tst.Sequential = true
	tst.Validate()

	trn.Init(0)
	tst.Init(0)

	// note: names must be in place when adding
	ss.Envs.Add(trn, tst)
}

// ConfigNet configures the network structure:
// a feedforward network with a 5x5 Input, two Hidden layers,
// and a 5x5 Output (Target) layer, fully connected.
func (ss *Sim) ConfigNet(net *egomlx.Network) {
	net.SetMaxData(ss.Config.Run.NData)
	net.SetRandSeed(ss.RandSeeds[0]) // init new separate random seed, using run = 0

	inp := net.AddLayer2D("Input", egomlx.Input, 5, 5)
	hid1 := net.AddLayer2D("Hidden1", egomlx.Hidden, ss.Config.Params.Hidden1Size.Y, ss.Config.Params.Hidden1Size.X)
	hid2 := net.AddLayer2D("Hidden2", egomlx.Hidden, ss.Config.Params.Hidden2Size.Y, ss.Config.Params.Hidden2Size.X)
	out := net.AddLayer2D("Output", egomlx.Target, 5, 5)

	// use this to position layers relative to each other
	// hid2.PlaceRightOf(hid1, 2)

	// note: see emergent/paths module for all the options on how to connect
	// NewFull returns a new paths.Full connectivity pattern
	full := paths.NewFull()

	net.ConnectLayers(inp, hid1, full, egomlx.Forward)
	net.ConnectLayers(hid1, hid2, full, egomlx.Forward)
	net.ConnectLayers(hid2, out, full, egomlx.Forward)

	err := net.Build()
	if errors.Log(err) != nil {
		return
	}
	net.Defaults()
	ss.ApplyParams()
	net.InitWeights()
}

// ApplyParams applies the network parameters, including any script.
func (ss *Sim) ApplyParams() {
	ss.Params.Script = ss.Config.Params.Script
	ss.Params.ApplyAll(ss.Net)
}

////////  Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	ss.Loops.ResetCounters()
	ss.SetRunName()
	ss.InitRandSeed(0)
	ss.ApplyParams()
	ss.StatsInit()
	ss.NewRun()
	ss.TrainUpdate.RecordSyns()
	ss.TrainUpdate.Update(Train, Trial)
}

// InitRandSeed initializes the random seed based on current training run number
func (ss *Sim) InitRandSeed(run int) {
	ss.RandSeeds.Set(run, ss.Net.Rand)
}

// NetViewUpdater returns the NetViewUpdate for given mode.
func (ss *Sim) NetViewUpdater(mode enums.Enum) *egomlx.NetViewUpdate {
	if mode.Int64() == Train.Int64() {
		return &ss.TrainUpdate
	}
	return &ss.TestUpdate
}

// ConfigLoops configures the control loops: Training, Testing
func (ss *Sim) ConfigLoops() {
	ls := looper.NewStacks()

	trials := int(math32.IntMultipleGE(float32(ss.Config.Run.Trials), float32(ss.Config.Run.NData)))
	cycles := ss.Config.Run.Cycles

	ls.AddStack(Train, Trial).
		AddLevel(Expt, 1).
		AddLevel(Run, ss.Config.Run.Runs).
		AddLevel(Epoch, ss.Config.Run.Epochs).
		AddLevelIncr(Trial, trials, ss.Config.Run.NData).
		AddLevel(Cycle, cycles)

	ls.AddStack(Test, Trial).
		AddLevel(Epoch, 1).
		AddLevelIncr(Trial, trials, ss.Config.Run.NData).
		AddLevel(Cycle, cycles)

	egomlx.LooperStandard(ls, ss.Net, ss.NetViewUpdater, Cycle, Trial, Train)

	ls.Stacks[Train].OnInit.Add("Init", ss.Init)

	ls.AddOnStartToLoop(Trial, "ApplyInputs", func(mode enums.Enum) {
		ss.ApplyInputs(mode.(Modes))
	})

	ls.Loop(Train, Run).OnStart.Add("NewRun", ss.NewRun)

	trainEpoch := ls.Loop(Train, Epoch)
	trainEpoch.IsDone.AddBool("NZeroStop", func() bool {
		stopNz := ss.Config.Run.NZero
		if stopNz <= 0 {
			return false
		}
		curModeDir := ss.Current.Dir(Train.String())
		// the NZero stat only exists once it has been computed;
		// guard against it not being present yet (e.g., before the
		// first epoch completes).
		if curModeDir.Node("NZero") == nil {
			return false
		}
		curNZero := int(curModeDir.Value("NZero").Float1D(-1))
		return curNZero >= stopNz
	})

	trainEpoch.OnStart.Add("TestAtInterval", func() {
		if (ss.Config.Run.TestInterval > 0) && ((trainEpoch.Counter.Cur+1)%ss.Config.Run.TestInterval == 0) {
			ss.TestAll()
		}
	})

	ls.AddOnStartToAll("StatsStart", ss.StatsStart)
	ls.AddOnEndToAll("StatsStep", ss.StatsStep)

	ls.Loop(Train, Run).OnEnd.Add("SaveWeights", func() {
		if !ss.Config.Log.SaveWeights {
			return
		}
		ctrString := fmt.Sprintf("%03d_%05d", ls.Loop(Train, Run).Counter.Cur, ls.Loop(Train, Epoch).Counter.Cur)
		fn := core.Filename("weights_" + ctrString + "_" + ss.RunName() + ".json")
		if err := ss.Net.SaveWeightsJSON(fn); errors.Log(err) != nil {
			return
		}
		mpi.Printf("Saving weights to %s\n", fn)
	})

	if ss.Config.GUI {
		egomlx.LooperUpdateNetView(ls, Cycle, Trial, ss.NetViewUpdater)

		ls.Stacks[Train].OnInit.Add("GUI-Init", ss.GUI.UpdateWindow)
		ls.Stacks[Test].OnInit.Add("GUI-Init", ss.GUI.UpdateWindow)
	}

	if ss.Config.Debug {
		mpi.Println(ls.DocString())
	}
	ss.Loops = ls
}

// ApplyInputs applies input patterns from given environment for given mode,
// for all the data-parallel items, using the standard start-of-trial logic.
func (ss *Sim) ApplyInputs(mode Modes) {
	net := ss.Net
	ndata := net.NData
	curModeDir := ss.Current.Dir(mode.String())
	ev := ss.Envs.ByMode(mode)
	lays := net.LayersByType(egomlx.Input, egomlx.Target)
	net.InitExt()
	for di := range ndata {
		ev.Step()
		curModeDir.StringValue("TrialName", ndata).SetString1D(ev.String(), di)
		for _, lnm := range lays {
			ly := net.LayerByName(lnm)
			st := ev.State(lnm)
			if st != nil {
				ly.ApplyExt(di, st)
			}
		}
	}
	net.ApplyExts()
}

// NewRun initializes a new Run level of the model.
func (ss *Sim) NewRun() {
	run := ss.Loops.Loop(Train, Run).Counter.Cur
	ss.InitRandSeed(run)
	ss.Envs.ByMode(Train).Init(run)
	ss.Envs.ByMode(Test).Init(run)
	ss.Net.InitWeights()
	if ss.Config.Run.StartWeights != "" {
		if err := ss.Net.OpenWeightsJSON(core.Filename(ss.Config.Run.StartWeights)); errors.Log(err) != nil {
			return
		}
		mpi.Printf("Starting with initial weights from: %s\n", ss.Config.Run.StartWeights)
	}
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	ss.Envs.ByMode(Test).Init(0)
	ss.Loops.ResetAndRun(Test)
	ss.Loops.Mode = Train // important because this is called from Train Run: go back.
}

////////  Inputs

// ConfigInputs generates the 25 random patterns and stores them,
// used for regenerating the embedded pattern set.
func (ss *Sim) ConfigInputs() {
	dt := table.New()
	metadata.SetName(dt, "Train")
	metadata.SetDoc(dt, "Training inputs")
	dt.AddStringColumn("Name")
	dt.AddFloat32Column("Input", 5, 5)
	dt.AddFloat32Column("Output", 5, 5)
	dt.SetNumRows(25)

	patterns.PermutedBinaryMinDiff(dt.Columns.Values[1], 6, 1, 0, 3)
	patterns.PermutedBinaryMinDiff(dt.Columns.Values[2], 6, 1, 0, 3)
	dt.SaveCSV("random_5x5_25_gen.tsv", tensor.Tab, table.Headers)

	tensorfs.DirFromTable(ss.Root.Dir("Inputs/Train"), dt)
}

// OpenTable opens a [table.Table] from embedded content, storing
// the data in the given tensorfs directory.
func (ss *Sim) OpenTable(dir *tensorfs.Node, fsys fs.FS, fnm, name, docs string) (*table.Table, error) {
	dt := table.New()
	metadata.SetName(dt, name)
	metadata.SetDoc(dt, docs)
	err := dt.OpenFS(embedfs, fnm, tensor.Tab)
	if errors.Log(err) != nil {
		return dt, err
	}
	tensorfs.DirFromTable(dir.Dir(name), dt)
	return dt, err
}

// OpenInputs opens the embedded random 5x5x25 patterns for the Train set.
func (ss *Sim) OpenInputs() {
	dir := ss.Root.Dir("Inputs")
	ss.OpenTable(dir, embedfs, "random_5x5_25.tsv", "Train", "Training inputs")
}

//////// Stats

// AddStatStd adds a standard stat compute function.
func (ss *Sim) AddStatStd(f func(mode enums.Enum, level enums.Enum, start bool)) {
	ss.StatFuncs = append(ss.StatFuncs, f)
}

// AddStat adds a custom stat compute function.
func (ss *Sim) AddStat(f func(mode Modes, level Levels, start bool)) {
	ss.AddStatStd(func(mode enums.Enum, level enums.Enum, start bool) {
		f(mode.(Modes), level.(Levels), start)
	})
}

// StatsStart is called by Looper at the start of given level, for each iteration.
// It needs to call RunStats Start at the next level down.
// e.g., each Epoch is the start of the full set of Trial Steps.
func (ss *Sim) StatsStart(lmd, ltm enums.Enum) {
	mode := lmd.(Modes)
	level := ltm.(Levels)
	if level <= Trial {
		return
	}
	ss.RunStats(mode, level-1, Start)
}

// StatsStep is called by Looper at each step of iteration,
// where it accumulates the stat results.
func (ss *Sim) StatsStep(lmd, ltm enums.Enum) {
	mode := lmd.(Modes)
	level := ltm.(Levels)
	if level == Cycle {
		return
	}
	ss.RunStats(mode, level, Step)
	tensorfs.DirTable(egomlx.StatsNode(ss.Stats, mode, level), nil).WriteToLog()
}

// RunStats runs the StatFuncs for given mode, level and phase.
func (ss *Sim) RunStats(mode Modes, level Levels, start bool) {
	for _, sf := range ss.StatFuncs {
		sf(mode, level, start)
	}
	if !start && ss.GUI.Tabs != nil {
		nm := mode.String() + " " + level.String() + " Plot"
		ss.GUI.Tabs.AsLab().GoUpdatePlot(nm)
	}
}

// SetRunName sets the overall run name, used for naming output logs and weight files
// based on params extra sheets and tag, and starting run number (for distributed runs).
func (ss *Sim) SetRunName() string {
	runName := ss.Params.RunName(ss.Config.Run.Run)
	ss.Current.StringValue("RunName", 1).SetString1D(runName, 0)
	return runName
}

// RunName returns the overall run name, used for naming output logs and weight files.
func (ss *Sim) RunName() string {
	return ss.Current.StringValue("RunName", 1).String1D(0)
}

// StatsInit initializes all the stats by calling Start across all modes and levels,
// and makes the standard plot tabs for the run stats.
func (ss *Sim) StatsInit() {
	for md, st := range ss.Loops.Stacks {
		mode := md.(Modes)
		for _, lev := range st.Order {
			level := lev.(Levels)
			if level == Cycle {
				continue
			}
			ss.RunStats(mode, level, Start)
		}
	}
	if ss.GUI.Tabs != nil {
		tbs := ss.GUI.Tabs.AsLab()
		_, idx := tbs.CurrentTab()
		tbs.PlotTensorFS(egomlx.StatsNode(ss.Stats, Train, Epoch))
		tbs.PlotTensorFS(egomlx.StatsNode(ss.Stats, Train, Run))
		tbs.PlotTensorFS(egomlx.StatsNode(ss.Stats, Test, Trial))
		tbs.SelectTabIndex(idx)
	}
}

// ConfigStats configures the functions that do all the stats computation
// in the tensorfs system, using the standard egomlx stats for counters,
// names and timing, plus the output error statistics.
// The output error is live -- Err tracks the network's current output
// error, and falls as the network learns.
func (ss *Sim) ConfigStats() {
	ss.Stats = ss.Root.Dir("Stats")
	ss.Current = ss.Stats.Dir("Current")

	ss.SetRunName()

	// last arg(s) are levels to exclude
	ss.AddStatStd(egomlx.StatLoopCounters(ss.Stats, ss.Current, ss.Loops, ss.Net, Trial, Cycle))
	ss.AddStatStd(egomlx.StatRunName(ss.Stats, ss.Current, ss.Loops, ss.Net, Trial, Cycle))
	ss.AddStatStd(egomlx.StatTrialName(ss.Stats, ss.Current, ss.Loops, ss.Net, Trial))
	ss.AddStatStd(egomlx.StatPerTrialMSec(ss.Stats, Train, Trial))
	ss.AddStat(ss.StatErr)
}

// StatErr computes the output layer error statistics:
// the per-item root-mean-square error between the Output layer
// activations and targets at the Trial level, with hierarchical
// reductions at higher levels, plus the standard zero-error epoch
// tracking stats. This is live -- Err tracks the network's current
// output error, and falls as the network learns.
func (ss *Sim) StatErr(mode Modes, level Levels, start bool) {
	statNames := []string{"Err", "NZero", "FirstZero", "LastZero"}
	statDocs := map[string]string{
		"Err":       "Root-mean-square error between the output activations and targets: per data item at the Trial level, mean over the epoch at the Epoch level, and final epoch value at higher levels. This falls as the network learns.",
		"NZero":     "The number of near-zero-error epochs in a row (Err < 0.001), for stopping training. Only recorded at the Epoch level and above, in Train mode.",
		"FirstZero": "The first epoch when the error was near zero.",
		"LastZero":  "The epoch when training was stopped because NZero got above the threshold for number of near-perfect epochs in a row.",
	}
	for _, name := range statNames {
		if name == "NZero" && (mode != Train || level == Trial) {
			continue
		}
		modeDir := ss.Stats.Dir(mode.String())
		curModeDir := ss.Current.Dir(mode.String())
		levelDir := modeDir.Dir(level.String())
		subDir := modeDir.Dir((level - 1).String())
		tsr := levelDir.Float64(name)
		ndata := ss.Net.NData
		if start {
			tsr.SetNumRows(0)
			plot.SetFirstStyler(tsr, func(s *plot.Style) {
				s.Range.SetMin(0)
				s.On = true
				switch name {
				case "NZero":
					s.On = false
				case "FirstZero", "LastZero":
					if level < Run {
						s.On = false
					}
				}
			})
			metadata.SetDoc(tsr, statDocs[name])
			switch name {
			case "NZero":
				if level == Epoch {
					curModeDir.Float64(name, 1).SetFloat1D(0, 0)
				}
			case "FirstZero", "LastZero":
				if level == Epoch {
					curModeDir.Float64(name, 1).SetFloat1D(-1, 0)
				}
			}
			continue
		}
		switch level {
		case Trial:
			if name != "Err" {
				tsr.AppendRowFloat(math.NaN())
				continue
			}
			for di := range ndata {
				rms := egomlx.StatRMSE(ss.Net, "Output", di, ss.Config.Run.UnitErrThr)
				curModeDir.Float64(name, ndata).SetFloat1D(rms, di)
				tsr.AppendRowFloat(rms)
			}
		case Epoch:
			switch name {
			case "NZero":
				errSum := stats.StatSum.Call(subDir.Value("Err")).Float1D(0)
				nz := curModeDir.Float64(name, 1).Float1D(0)
				if errSum < 0.001 {
					nz++
				} else {
					nz = 0
				}
				curModeDir.Float64(name, 1).SetFloat1D(nz, 0)
				tsr.AppendRowFloat(nz)
			case "FirstZero":
				stat := curModeDir.Float64(name, 1).Float1D(0)
				nz := curModeDir.Float64("NZero", 1).Float1D(0)
				if stat < 0 && nz == 1 {
					stat = curModeDir.Int("Epoch", 1).Float1D(0)
				}
				curModeDir.Float64(name, 1).SetFloat1D(stat, 0)
				tsr.AppendRowFloat(stat)
			case "LastZero":
				stat := curModeDir.Float64(name, 1).Float1D(0)
				nz := curModeDir.Float64("NZero", 1).Float1D(0)
				if stat < 0 && nz >= float64(ss.Config.Run.NZero) {
					stat = curModeDir.Int("Epoch", 1).Float1D(0)
				}
				curModeDir.Float64(name, 1).SetFloat1D(stat, 0)
				tsr.AppendRowFloat(stat)
			default: // Err
				tsr.AppendRowFloat(stats.StatMean.Call(subDir.Value(name)).Float1D(0))
			}
		case Run:
			tsr.AppendRowFloat(stats.StatFinal.Call(subDir.Value(name)).Float1D(0))
		default: // Expt
			tsr.AppendRowFloat(stats.StatMean.Call(subDir.Value(name)).Float1D(0))
		}
	}
}

// StatCounters returns counters string to show at bottom of netview.
func (ss *Sim) StatCounters(mode, level enums.Enum) string {
	counters := ss.Loops.Stacks[mode].CountersString()
	vu := ss.NetViewUpdater(mode)
	if vu == nil || vu.View == nil {
		return counters
	}
	di := vu.View.Di
	counters += fmt.Sprintf(" Di: %d", di)
	curModeDir := ss.Current.Dir(mode.String())
	if curModeDir.Node("TrialName") == nil {
		return counters
	}
	counters += fmt.Sprintf(" TrialName: %s", curModeDir.StringValue("TrialName").String1D(di))
	if curModeDir.Node("Err") != nil {
		counters += fmt.Sprintf(" Err: %.4g", curModeDir.Float64("Err").Float1D(di))
	}
	return counters
}

//////// GUI

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI(b tree.Node) {
	ss.GUI.MakeBody(b, ss, ss.Root, ss.Config.Name, ss.Config.Title, ss.Config.Doc)
	ss.GUI.StopLevel = Trial
	nv := ss.GUI.AddNetView("Network")
	nv.Settings.MaxRecs = 2 * ss.Config.Run.Cycles
	nv.Settings.Raster.Max = ss.Config.Run.Cycles
	nv.SetNet(ss.Net)
	ss.TrainUpdate.Config(nv, egomlx.Trial, ss.StatCounters)
	ss.TestUpdate.Config(nv, egomlx.Trial, ss.StatCounters)
	ss.GUI.OnStop = func(mode, level enums.Enum) {
		vu := ss.NetViewUpdater(mode)
		vu.UpdateWhenStopped(mode, level)
	}

	nv.SceneXYZ().Camera.Pose.Pos.Set(0, 1, 2.75)
	nv.SceneXYZ().Camera.LookAt(math32.Vec3(0, 0, 0), math32.Vec3(0, 1, 0))

	ss.StatsInit()
	ss.GUI.FinalizeGUI(false)
}

// MakeToolbar makes the custom toolbar items.
func (ss *Sim) MakeToolbar(p *tree.Plan) {
	ss.GUI.AddLooperCtrl(p, ss.Loops)

	tree.Add(p, func(w *core.Separator) {})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{
		Label:   "New Seed",
		Icon:    icons.Add,
		Tooltip: "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time.",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.RandSeeds.NewSeeds()
		},
	})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{
		Label:   "README",
		Icon:    icons.FileMarkdown,
		Tooltip: "Opens your browser on the README file that contains instructions for how to run this model.",
		Active:  egui.ActiveAlways,
		Func: func() {
			core.TheApp.OpenURL(ss.Config.URL)
		},
	})
}

// RunNoGUI runs the sim without a GUI, in a headless / terminal mode.
func (ss *Sim) RunNoGUI() {
	ss.Init()

	if ss.Config.Params.Note != "" {
		mpi.Printf("Note: %s\n", ss.Config.Params.Note)
	}
	if ss.Config.Log.SaveWeights {
		mpi.Printf("Saving final weights per run\n")
	}

	runName := ss.SetRunName()
	netName := ss.Net.Name
	cfg := &ss.Config.Log
	egomlx.OpenLogFiles(ss.Loops, ss.Stats, netName, runName, [][]string{cfg.Train, cfg.Test})

	mpi.Printf("Running %d Runs starting at %d: %s\n", ss.Config.Run.Runs, ss.Config.Run.Run, runName)
	ss.Loops.Loop(Train, Run).Counter.SetCurMaxPlusN(ss.Config.Run.Run, ss.Config.Run.Runs)
	ss.Loops.Run(Train)

	egomlx.CloseLogFiles(ss.Loops, ss.Stats, Cycle)
}
