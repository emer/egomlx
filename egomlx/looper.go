// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package egomlx

import (
	"cogentcore.org/core/enums"
	"github.com/emer/emergent/v2/looper"
	"github.com/emer/emergent/v2/netview"
)

// ViewTimes are the options for when the NetView can be updated.
// For feedforward backprop models, the natural timescale is Trial,
// with each trial being a single forward (and, when training,
// backward) pass.
type ViewTimes int32 //enums:enum

const (
	// Cycle is an update of the network state at the cycle level,
	// which for feedforward models is typically the same as Trial
	// (one pass per trial).
	Cycle ViewTimes = iota

	// Trial is an update at the trial level, the standard
	// timescale for feedforward backprop models.
	Trial
)

// ViewTimeCycles are the cycle intervals associated with each ViewTimes level.
var ViewTimeCycles = []int{1, 1}

// Cycles returns the number of cycles associated with a given view time.
func (vt ViewTimes) Cycles() int {
	if int(vt) >= len(ViewTimeCycles) {
		return 1
	}
	return ViewTimeCycles[vt]
}

// NetViewUpdate manages time scales for updating the [netview] NetView.
// Use one of these for each mode you want to control separately.
type NetViewUpdate struct {

	// On toggles update of display on.
	On bool

	// Time scale to update the network view (Cycle to Trial timescales).
	Time ViewTimes

	// CounterFunc returns the counter string showing current counters etc.
	CounterFunc func(mode, level enums.Enum) string `display:"-"`

	// View is the network view.
	View *netview.NetView `display:"-"`
}

// Config configures for given [netview] [netview.NetView], time and counter function,
// which returns a string to show at the bottom of the netview,
// given the current mode and level.
func (vu *NetViewUpdate) Config(nv *netview.NetView, tm ViewTimes, fun func(mode, level enums.Enum) string) {
	vu.View = nv
	vu.On = true
	vu.Time = tm
	vu.CounterFunc = fun
}

// ShouldUpdate returns true if the view is On,
// View is != nil, and it is visible.
func (vu *NetViewUpdate) ShouldUpdate() bool {
	if !vu.On || vu.View == nil || !vu.View.IsVisible() {
		return false
	}
	return true
}

// GoUpdate does an update if view is On, visible and active,
// including recording new data and driving update of display.
// This version is only for calling from a separate goroutine,
// not the main event loop (see also Update).
func (vu *NetViewUpdate) GoUpdate(mode, level enums.Enum) {
	if !vu.ShouldUpdate() {
		return
	}
	counters := vu.CounterFunc(mode, level)
	vu.View.Record(counters, -1) // -1 = default incrementing raster
	vu.View.GoUpdateView()
}

// Update does an update if view is On, visible and active,
// including recording new data and driving update of display.
// This version is only for calling from the main event loop
// (see also GoUpdate).
func (vu *NetViewUpdate) Update(mode, level enums.Enum) {
	if !vu.ShouldUpdate() {
		return
	}
	counters := vu.CounterFunc(mode, level)
	vu.View.Record(counters, -1) // -1 = use a dummy counter
	vu.View.UpdateView()
}

// UpdateWhenStopped does an update when the network updating was stopped
// either via stepping or hitting the stop button.
// This is only for calling from a separate goroutine,
// not the main event loop.
func (vu *NetViewUpdate) UpdateWhenStopped(mode, level enums.Enum) {
	if !vu.ShouldUpdate() {
		return
	}
	counters := vu.CounterFunc(mode, level)
	vu.View.Record(counters, -1) // -1 = use a dummy counter
	vu.View.GoUpdateView()
}

// IsViewingSynapse returns true if netview is actively viewing synapses.
func (vu *NetViewUpdate) IsViewingSynapse() bool {
	if !vu.ShouldUpdate() {
		return false
	}
	return vu.View.IsViewingSynapse()
}

// UpdateCycle triggers an update at the Cycle timescale,
// using the current counter values to display at the bottom of the view.
func (vu *NetViewUpdate) UpdateCycle(cyc int, mode, level enums.Enum) {
	if !vu.ShouldUpdate() {
		return
	}
	if vu.Time == Trial { // only trial
		return
	}
	vtc := vu.Time.Cycles()
	if (cyc+1)%vtc == 0 {
		vu.GoUpdate(mode, level)
	}
}

// RecordSyns records synaptic data -- stored separate from unit data
// and only needs to be called when synaptic values are updated.
// NetView displays this recorded data when Update is next called.
func (vu *NetViewUpdate) RecordSyns() {
	if !vu.ShouldUpdate() {
		return
	}
	vu.View.RecordSyns()
}

// LooperUpdateNetView adds netview update calls to the given
// trial and cycle levels for each mode in the given looper Stacks,
// using the given viewFunc function to get the NetViewUpdate for
// each mode.
func LooperUpdateNetView(ls *looper.Stacks, cycle, trial enums.Enum, viewFunc func(mode enums.Enum) *NetViewUpdate) {
	for mode, st := range ls.Stacks {
		viewUpdt := viewFunc(mode)
		cycLoop := st.Loops[cycle]
		cycLoop.OnEnd.Add("GUI:UpdateNetView", func() {
			viewUpdt.UpdateCycle(cycLoop.Counter.Cur, mode, cycle)
		})
		trlLoop := st.Loops[trial]
		trlLoop.OnEnd.Add("GUI:UpdateNetView", func() {
			viewUpdt.GoUpdate(mode, trial)
		})
	}
}

// LooperStandard adds standard netview update and weight readback calls
// to the given looper Stacks, for the given cycle and trial levels.
// For feedforward backprop models, the trial OnEnd handler runs
// the forward pass (and, in training mode, the backward pass with
// optimizer weight updates), and then updates the netview.
// In training mode, DWt gradient values are read back only when the
// netview is actively viewing synapses, to avoid the extra cost
// otherwise: see [Network.Backward].
func LooperStandard(ls *looper.Stacks, nt *Network, viewFunc func(mode enums.Enum) *NetViewUpdate, cycle, trial, trainMode enums.Enum) {
	for mode, st := range ls.Stacks {
		testing := mode.Int64() != trainMode.Int64()
		trlLoop := st.Loops[trial]
		trlLoop.OnEnd.Add("Forward", func() {
			view := viewFunc(mode)
			viewing := view != nil && view.IsViewingSynapse()
			if testing {
				nt.TestTrial()
				if viewing {
					nt.ReadWeights()
					view.RecordSyns() // note: critical to update weights before record
				}
				return
			}
			dwts := DWtNone
			if viewing {
				dwts = DWtReadRaw
			}
			nt.TrainTrial(dwts)
			if viewing {
				// TrainTrial already read the updated weights if dwts
				// was DWtReadRaw, but do it again to be certain the
				// recorded values are current.
				nt.ReadWeights()
				view.RecordSyns() // note: critical to update weights before record
			}
		})
	}
	LooperUpdateNetView(ls, cycle, trial, viewFunc)
}
