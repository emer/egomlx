// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package egomlx

import (
	"strings"
	"time"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/base/metadata"
	"cogentcore.org/core/base/timer"
	"cogentcore.org/core/enums"
	"cogentcore.org/lab/plot"
	"cogentcore.org/lab/stats/stats"
	"cogentcore.org/lab/table"
	"cogentcore.org/lab/tensor"
	"cogentcore.org/lab/tensorfs"
	"github.com/emer/emergent/v2/looper"
)

// StatsNode returns tensorfs Dir Node for given mode, level.
// The standard stats directory organization is statsDir / mode / level,
// where the values are the enum String values (e.g., "Train/Trial").
func StatsNode(statsDir *tensorfs.Node, mode, level enums.Enum) *tensorfs.Node {
	modeDir := statsDir.Dir(mode.String())
	return modeDir.Dir(level.String())
}

// LogFilename returns a standard log file name as netName_runName_logName.tsv
func LogFilename(netName, runName, logName string) string {
	return netName + "_" + runName + "_" + logName + ".tsv"
}

// OpenLogFile, if on == true, sets the log file for given table using given
// netName, runName, and logName in order.
func OpenLogFile(on bool, dt *table.Table, netName, runName, logName string) {
	if !on {
		return
	}
	fnm := LogFilename(netName, runName, logName)
	tensor.SetPrecision(dt, 4)
	dt.OpenLog(fnm, tensor.Tab)
}

// OpenLogFiles opens the log files for modes and levels of the looper,
// based on the lists of level names, ordered by modes in numerical order.
// The netName and runName are used for naming the file, along with
// the mode_level in lower case.
// The tables are the [tensorfs.DirTable] of each stats directory,
// which caches the table on the directory node, so that the same
// table object (and its log file state) is used by the stats
// that write values into that directory.
func OpenLogFiles(ls *looper.Stacks, statsDir *tensorfs.Node, netName, runName string, modeLevels [][]string) {
	modes := ls.Modes()
	for i, mode := range modes {
		if i >= len(modeLevels) {
			return
		}
		levels := modeLevels[i]
		st := ls.Stacks[mode]
		for _, level := range st.Order {
			on := false
			for _, lev := range levels {
				if lev == level.String() {
					on = true
					break
				}
			}
			if !on {
				continue
			}
			logName := strings.ToLower(mode.String() + "_" + level.String())
			dt := tensorfs.DirTable(StatsNode(statsDir, mode, level), nil)
			fnm := LogFilename(netName, runName, logName)
			tensor.SetPrecision(dt, 4)
			dt.OpenLog(fnm, tensor.Tab)
		}
	}
}

// CloseLogFiles closes all the log files for each mode and level of the looper,
// excluding given level(s).
func CloseLogFiles(ls *looper.Stacks, statsDir *tensorfs.Node, exclude ...enums.Enum) {
	modes := ls.Modes() // mode enum order
	for _, mode := range modes {
		st := ls.Stacks[mode]
		for _, level := range st.Order {
			if StatExcludeLevel(level, exclude...) {
				continue
			}
			dt := tensorfs.DirTable(StatsNode(statsDir, mode, level), nil)
			dt.CloseLog()
		}
	}
}

// StatExcludeLevel returns true if given level is among the list of levels to exclude.
func StatExcludeLevel(level enums.Enum, exclude ...enums.Enum) bool {
	for _, ex := range exclude {
		if level == ex {
			return true
		}
	}
	return false
}

// StatLoopCounters adds the counters from each stack, loop level for given
// looper Stacks to the given tensorfs stats. This is typically the first
// Stat to add, so these counters will be used for X axis values.
// The stat is run with start = true before returning, so that the stats
// are already initialized first before anything else.
// The first mode's counters (typically Train) are automatically added to all
// subsequent modes so they automatically track training levels.
//
//   - currentDir is a tensorfs directory to store the current values of each counter.
//   - trialLevel is the Trial level enum, which automatically handles the
//     iteration over ndata parallel trials.
//   - exclude is a list of loop levels to exclude (e.g., Cycle).
func StatLoopCounters(statsDir, currentDir *tensorfs.Node, ls *looper.Stacks, net *Network, trialLevel enums.Enum, exclude ...enums.Enum) func(mode, level enums.Enum, start bool) {
	modes := ls.Modes() // mode enum order
	fun := func(mode, level enums.Enum, start bool) {
		for mi := range 2 {
			st := ls.Stacks[mode]
			prefix := ""
			if mi == 0 {
				if modes[mi].Int64() == mode.Int64() { // skip train in train..
					continue
				}
				ctrMode := modes[mi]
				st = ls.Stacks[ctrMode]
				prefix = ctrMode.String()
			}
			for _, lev := range st.Order {
				// don't record counter for levels above it
				if level.Int64() > lev.Int64() {
					continue
				}
				if StatExcludeLevel(lev, exclude...) {
					continue
				}
				name := prefix + lev.String() // name of stat = level
				ndata := net.NData
				modeDir := statsDir.Dir(mode.String())
				curModeDir := currentDir.Dir(mode.String())
				levelDir := modeDir.Dir(level.String())
				tsr := levelDir.Int(name)
				if start {
					tsr.SetNumRows(0)
					plot.SetFirstStyler(tsr, func(s *plot.Style) {
						s.Range.SetMin(0)
					})
					if level.Int64() == trialLevel.Int64() {
						for di := range ndata {
							curModeDir.Int(name, ndata).SetInt1D(0, di)
						}
					}
					metadata.SetDoc(tsr, "Loop counter for given level")
					continue
				}
				ctr := st.Loops[lev].Counter.Cur
				if level.Int64() == trialLevel.Int64() {
					for di := range ndata {
						curModeDir.Int(name, ndata).SetInt1D(ctr, di)
						tsr.AppendRowInt(ctr)
						if lev.Int64() == trialLevel.Int64() {
							ctr++
						}
					}
				} else {
					curModeDir.Int(name, 1).SetInt1D(ctr, 0)
					tsr.AppendRowInt(ctr)
				}
			}
		}
	}
	for _, md := range modes {
		st := ls.Stacks[md]
		for _, lev := range st.Order {
			if StatExcludeLevel(lev, exclude...) {
				continue
			}
			fun(md, lev, true)
		}
	}
	return fun
}

// StatRunName adds a "RunName" stat to every mode and level of looper,
// subject to exclusion list, which records the current value of the
// "RunName" string in currentDir, which identifies the parameters and tag
// for this run.
func StatRunName(statsDir, currentDir *tensorfs.Node, ls *looper.Stacks, net *Network, trialLevel enums.Enum, exclude ...enums.Enum) func(mode, level enums.Enum, start bool) {
	return func(mode, level enums.Enum, start bool) {
		name := "RunName"
		modeDir := statsDir.Dir(mode.String())
		levelDir := modeDir.Dir(level.String())
		tsr := levelDir.StringValue(name)
		ndata := net.NData
		runNm := currentDir.StringValue(name, 1).String1D(0)

		if start {
			tsr.SetNumRows(0)
			metadata.SetDoc(tsr, "Identifies the current parameters and tag label for current runs, can be used to sort different runs during interactive exploration")
			return
		}
		if level.Int64() == trialLevel.Int64() {
			for range ndata {
				tsr.AppendRowString(runNm)
			}
		} else {
			tsr.AppendRowString(runNm)
		}
	}
}

// StatTrialName adds a "TrialName" stat to the given Trial level in every mode of looper,
// which records the current value of the "TrialName" string in currentDir, which
// contains a string description of the current trial.
func StatTrialName(statsDir, currentDir *tensorfs.Node, ls *looper.Stacks, net *Network, trialLevel enums.Enum) func(mode, level enums.Enum, start bool) {
	return func(mode, level enums.Enum, start bool) {
		if level.Int64() != trialLevel.Int64() {
			return
		}
		name := "TrialName"
		modeDir := statsDir.Dir(mode.String())
		curModeDir := currentDir.Dir(mode.String())
		levelDir := modeDir.Dir(level.String())
		tsr := levelDir.StringValue(name)
		ndata := net.NData
		if start {
			tsr.SetNumRows(0)
			metadata.SetDoc(tsr, "Name of the current trial, generated by the environment to describe current input patterns")
			return
		}
		for di := range ndata {
			trlNm := curModeDir.StringValue(name, ndata).String1D(di)
			tsr.AppendRowString(trlNm)
		}
	}
}

// StatPerTrialMSec returns a Stats function that reports the number of milliseconds
// per trial, for the given levels and training mode enum values.
// Stats will be recorded at levels above the given trial level.
func StatPerTrialMSec(statsDir *tensorfs.Node, trainMode enums.Enum, trialLevel enums.Enum) func(mode, level enums.Enum, start bool) {
	var epcTimer timer.Time
	levels := make([]enums.Enum, 10) // should be enough
	levels[0] = trialLevel
	return func(mode, level enums.Enum, start bool) {
		levi := int(level.Int64() - trialLevel.Int64())
		if mode.Int64() != trainMode.Int64() || levi <= 0 {
			return
		}
		levels[levi] = level
		name := "PerTrialMSec"
		modeDir := statsDir.Dir(mode.String())
		levelDir := modeDir.Dir(level.String())
		tsr := levelDir.Float64(name)
		if start {
			tsr.SetNumRows(0)
			plot.SetFirstStyler(tsr, func(s *plot.Style) {
				s.Range.SetMin(0)
			})
			metadata.SetDoc(tsr, "Milliseconds per trial of wall-clock time, averaged over an epoch, to provide computational timing info")
			return
		}
		switch levi {
		case 1:
			epcTimer.Stop()
			subDir := modeDir.Dir(levels[0].String())
			trls := errors.Ignore1(subDir.Values())[0] // must be a stat
			epcTimer.N = trls.Len()
			pertrl := float64(epcTimer.Avg()) / float64(time.Millisecond)
			tsr.AppendRowFloat(pertrl)
			epcTimer.ResetStart()
		default:
			subDir := modeDir.Dir(levels[levi-1].String())
			tsr.AppendRow(stats.StatMean.Call(subDir.Value(name)))
		}
	}
}
