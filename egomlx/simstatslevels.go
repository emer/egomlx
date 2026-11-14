// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package egomlx

import (
	"cogentcore.org/core/base/metadata"
	"cogentcore.org/core/enums"
	"cogentcore.org/lab/matrix"
	"cogentcore.org/lab/plot"
	"cogentcore.org/lab/stats/metric"
	"cogentcore.org/lab/stats/stats"
	"cogentcore.org/lab/tensor"
	"cogentcore.org/lab/tensorfs"
)

// StatLevelAll returns a Stats function that copies stats from given mode
// and level, without resetting at the start, to accumulate all rows
// over time until reset manually. The styleFunc, if non-nil, does plot styling
// based on the current column. Typically used to accumulate the Run level
// stats across all runs, in a "RunAll" directory.
func StatLevelAll(statsDir *tensorfs.Node, srcMode, srcLevel enums.Enum, styleFunc func(s *plot.Style, col tensor.Values)) func(mode, level enums.Enum, start bool) {
	return func(mode, level enums.Enum, start bool) {
		if srcMode.Int64() != mode.Int64() || srcLevel.Int64() != level.Int64() {
			return
		}
		modeDir := statsDir.Dir(mode.String())
		levelDir := modeDir.Dir(level.String())
		allDir := modeDir.Dir(level.String() + "All")
		cols := levelDir.NodesFunc(nil) // all nodes
		for _, cl := range cols {
			clv := cl.Tensor.(tensor.Values)
			if clv.NumDims() == 0 || clv.DimSize(0) == 0 {
				continue
			}
			if start {
				trg := tensorfs.ValueType(allDir, cl.Name(), clv.DataType(), clv.ShapeSizes()...)
				if trg.Len() == 0 {
					if styleFunc != nil {
						plot.SetFirstStyler(trg, func(s *plot.Style) {
							styleFunc(s, clv)
						})
					}
					trg.SetNumRows(0)
				}
			} else {
				trg := tensorfs.ValueType(allDir, cl.Name(), clv.DataType())
				trg.AppendRow(clv.RowTensor(clv.DimSize(0) - 1))
			}
		}
	}
}

// StatCopyLevel returns a Stats function that copies given stat names from given
// source mode and level, into given target mode at the same level, with the
// source mode name as a prefix (e.g., Test_CorSim in the Train Epoch log).
// The styleFunc, if non-nil, does plot styling based on the current column name.
// Due to asymmetries in timing for when stats are made, this assumes a Float64
// scalar stat value. runLevel is used for getting a final value, else an average.
func StatCopyLevel(statsDir *tensorfs.Node, trgMode, srcMode, srcLevel, runLevel enums.Enum, styleFunc func(s *plot.Style, stName string), statNames ...string) func(mode, level enums.Enum, start bool) {
	levels := make([]enums.Enum, 10) // should be enough
	return func(mode, level enums.Enum, start bool) {
		levi := int(level.Int64() - srcLevel.Int64())
		if trgMode.Int64() != mode.Int64() || levi < 0 {
			return
		}
		levels[levi] = level
		modeDir := statsDir.Dir(mode.String())
		levelDir := modeDir.Dir(level.String())

		srcModeDir := statsDir.Dir(srcMode.String())
		srcLevelDir := srcModeDir.Dir(srcLevel.String())
		for _, snm := range statNames {
			name := srcMode.String() + "_" + snm
			tsr := levelDir.Float64(name)
			if start {
				if tsr.Len() == 0 {
					if styleFunc != nil {
						plot.SetFirstStyler(tsr, func(s *plot.Style) {
							styleFunc(s, snm)
						})
					}
				}
				tsr.SetNumRows(0)
				continue
			}
			switch levi {
			case 0:
				cl := srcLevelDir.Node(snm)
				if cl == nil || cl.Tensor == nil {
					continue
				}
				clv := cl.Tensor.(tensor.Values)
				if clv.NumDims() == 0 || clv.DimSize(0) == 0 {
					continue
				}
				tsr.AppendRow(clv.RowTensor(clv.DimSize(0) - 1))
			case int(runLevel.Int64() - srcLevel.Int64()):
				subDir := modeDir.Dir(levels[levi-1].String())
				tsr.AppendRow(stats.StatFinal.Call(subDir.Value(name)))
			default:
				subDir := modeDir.Dir(levels[levi-1].String())
				tsr.AppendRow(stats.StatMean.Call(subDir.Value(name)))
			}
		}
	}
}

// StatLayerAct returns a Stats function that records the average and
// maximum Act values for given layers, as LayerName_ActAvg and
// LayerName_ActMax stats, for the given trainMode only.
// At the trial level, values are computed per data-parallel item,
// with higher levels computing the Mean of the lower level, except
// for the runLevel, which records the Final value.
// This is the feedforward analog of the axon StatLayerActGe stats,
// useful for monitoring dead / saturated units in hidden layers.
func StatLayerAct(statsDir *tensorfs.Node, net *Network, trainMode, trialLevel, runLevel enums.Enum, layerNames ...string) func(mode, level enums.Enum, start bool) {
	statNames := []string{"ActAvg", "ActMax"}
	statDocs := map[string]string{
		"ActAvg": "Average activation across all units in the layer.",
		"ActMax": "Maximum activation across all units in the layer.",
	}
	levels := make([]enums.Enum, 10) // should be enough
	levels[0] = trialLevel
	return func(mode, level enums.Enum, start bool) {
		levi := int(level.Int64() - trialLevel.Int64())
		if mode.Int64() != trainMode.Int64() || levi < 0 {
			return
		}
		levels[levi] = level
		modeDir := statsDir.Dir(mode.String())
		levelDir := modeDir.Dir(level.String())
		ndata := net.NData
		for _, lnm := range layerNames {
			ly := net.LayerByName(lnm)
			if ly == nil {
				continue
			}
			for si, statName := range statNames {
				name := lnm + "_" + statName
				tsr := levelDir.Float64(name)
				if start {
					tsr.SetNumRows(0)
					plot.SetFirstStyler(tsr, func(s *plot.Style) {
						s.Range.SetMin(0)
					})
					metadata.SetDoc(tsr, statDocs[statName])
					continue
				}
				switch levi {
				case 0:
					for di := range ndata {
						avg, mx := ly.ActAvgMax(di, "Act")
						stat := avg
						if si == 1 {
							stat = mx
						}
						tsr.AppendRowFloat(float64(stat))
					}
				case int(runLevel.Int64() - trialLevel.Int64()):
					subDir := modeDir.Dir(levels[levi-1].String())
					tsr.AppendRow(stats.StatFinal.Call(subDir.Value(name)))
				default:
					subDir := modeDir.Dir(levels[levi-1].String())
					tsr.AppendRow(stats.StatMean.Call(subDir.Value(name)))
				}
			}
		}
	}
}

// PCAStrongThr is the threshold for counting PCA eigenvalues as "strong".
var PCAStrongThr = 0.01

// StatPCA returns a Stats function that computes PCA NStrong, Top5, Next5, and Rest
// stats on the Act values of the given layers, which are important for tracking
// hogging dynamics where the representational space is not efficiently distributed.
// Uses Sample units for layers, and SVD computation is reasonably efficient.
// It only runs for given trainMode, from given Trial level upward,
// with higher levels computing the Mean of lower levels.
// The trial level just records Act values for layers in a separate PCA subdir,
// which are input to the next level computation where PCA is computed,
// every interval epochs (passed as epc to the returned function).
func StatPCA(statsDir, currentDir *tensorfs.Node, net *Network, interval int, trainMode, trialLevel, runLevel enums.Enum, layerNames ...string) func(mode, level enums.Enum, start bool, epc int) {
	statNames := []string{"PCA_NStrong", "PCA_Top5", "PCA_Next", "PCA_Rest"}
	statDocs := map[string]string{
		"PCA_NStrong": "The number of PCA components in the covariance matrix of layer activity across the epoch, which have an eigenvalue above the PCAStrongThr (0.01 default), indicating that a component is making at least some kind of reasonable contribution to the overall representational space. In general this indicates how much `information' present in the layer's representations, with higher values generally being better.",
		"PCA_Top5":    "The summed strength of the eigenvalues for the top 5 PCA components in the covariance matrix of layer activity across the epoch. If these are very strong, it can indicate that the layer's activity is insufficiently diverse, and is being dominated by a few strong representations (i.e., `hog' units). See PCA_Next and PCA_Rest for comparisons; These values are only meaningful in evaluating the effects of different parameters that alter the balance of these factors: absolute values are not well defined.",
		"PCA_Next":    "The summed strength of the next 5 eigenvalues (see PCA_Top5). The relative strengths of this compared with Top5 gives a better sense of the balance vs. `hog' unit dynamic.",
		"PCA_Rest":    "The summed strength of the remaining eigenvalues beyond the top 10, which is useful in the same way PCA_Next is relative to PCA_Top5.",
	}
	levels := make([]enums.Enum, 10) // should be enough
	return func(mode, level enums.Enum, start bool, epc int) {
		levi := int(level.Int64() - trialLevel.Int64())
		if mode.Int64() != trainMode.Int64() || levi < 0 {
			return
		}
		levels[levi] = level
		modeDir := statsDir.Dir(mode.String())
		curModeDir := currentDir.Dir(mode.String())
		curPCADir := curModeDir.Dir("PCA")
		pcaDir := statsDir.Dir("PCA")
		levelDir := modeDir.Dir(level.String())
		ndata := net.NData
		for _, lnm := range layerNames {
			ly := net.LayerByName(lnm)
			if ly == nil {
				continue
			}
			sizes := []int{ndata}
			sizes = append(sizes, ly.GetSampleShape().Sizes...)
			vtsr := pcaDir.Float64(lnm, sizes...)
			if levi == 0 {
				ltsr := curPCADir.Float64(lnm+"_Act", ly.GetSampleShape().Sizes...)
				if start {
					vtsr.SetNumRows(0)
				} else {
					for di := range ndata {
						ly.UnitValuesSampleTensor(ltsr, "Act", di)
						vtsr.AppendRow(ltsr)
					}
				}
				continue
			}
			var svals [4]float64 // in statNames order
			hasNew := false
			if !start && levi == 1 {
				if interval > 0 && epc%interval == 0 && vtsr.DimSize(0) > 1 {
					hasNew = true
					vals := curPCADir.Float64("Vals_" + lnm)
					covar := curPCADir.Float64("Covar_" + lnm)
					metric.CovarianceMatrixOut(metric.Covariance, vtsr, covar)
					matrix.SVDValuesOut(covar, vals)
					ln := vals.Len()
					for i := range ln {
						v := vals.Float1D(i)
						if v < PCAStrongThr {
							svals[0] = float64(i)
							break
						}
					}
					for i := range 5 {
						if ln >= 5 {
							svals[1] += vals.Float1D(i)
						}
						if ln >= 10 {
							svals[2] += vals.Float1D(i + 5)
						}
					}
					svals[1] /= 5
					svals[2] /= 5
					if ln > 10 {
						sum := stats.Sum(vals).Float1D(0)
						svals[3] = (sum - (svals[1] + svals[2])) / float64(ln-10)
					}
				}
			}
			for si, statName := range statNames {
				name := lnm + "_" + statName
				tsr := levelDir.Float64(name)
				if start {
					tsr.SetNumRows(0)
					plot.SetFirstStyler(tsr, func(s *plot.Style) {
						s.Range.SetMin(0)
					})
					metadata.SetDoc(tsr, statDocs[statName])
					continue
				}
				switch levi {
				case 1:
					var stat float64
					nr := tsr.DimSize(0)
					if nr > 0 {
						stat = tsr.FloatRow(nr-1, 0)
					}
					if hasNew {
						stat = svals[si]
					}
					tsr.AppendRowFloat(stat)
				case int(runLevel.Int64() - trialLevel.Int64()):
					subDir := modeDir.Dir(levels[levi-1].String())
					tsr.AppendRow(stats.StatFinal.Call(subDir.Value(name)))
				default:
					subDir := modeDir.Dir(levels[levi-1].String())
					tsr.AppendRow(stats.StatMean.Call(subDir.Value(name)))
				}
			}
		}
	}
}
