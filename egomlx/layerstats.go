// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package egomlx

import (
	"cogentcore.org/core/math32"
)

// unitVals returns the values of the given unit variable for all units
// in the layer at data-parallel index di, as a slice indexed by the
// 1D unit index. Returns nil if the variable is not found.
func (ly *Layer) unitVals(varNm string, di int) []float32 {
	vi, err := NeuronVarIndexByName(varNm)
	if err != nil {
		return nil
	}
	nu := ly.Shape.Len()
	vals := make([]float32, nu)
	for ui := range nu {
		vals[ui] = ly.UnitValue1D(vi, ui, di)
	}
	return vals
}

// Correlation computes the correlation (-1, 1) between two unit
// variables (e.g., "Targ" and "Act") across all units in the layer,
// for data-parallel index di. Correlation is the centered cosine
// (normalized dot product); center = false computes the cosine instead.
// This is a good continuous, normalized measure of output performance
// (1 = patterns are identical up to scaling).
// Returns 0 if either variable has zero variance.
func (ly *Layer) Correlation(di int, aVar, bVar string, center bool) float32 {
	av := ly.unitVals(aVar, di)
	bv := ly.unitVals(bVar, di)
	if av == nil || bv == nil {
		return math32.NaN()
	}
	nu := len(av)
	avgA, avgB := float32(0), float32(0)
	if center && nu > 0 {
		for ui := range nu {
			avgA += av[ui]
			avgB += bv[ui]
		}
		avgA /= float32(nu)
		avgB /= float32(nu)
	}
	cosv, ssa, ssb := float32(0), float32(0), float32(0)
	for ui := range nu {
		a := av[ui] - avgA
		b := bv[ui] - avgB
		cosv += a * b
		ssa += a * a
		ssb += b * b
	}
	dist := math32.Sqrt(ssa * ssb)
	if dist != 0 {
		cosv /= dist
	}
	return cosv
}

// TargetOnOff returns the extent to which, according to the target
// variable (e.g., "Targ"), units were active (actVar, e.g., "Act") when
// they should have been, and inactive where they should have been,
// for data-parallel index di.
// onWasOff = proportion of target-on units that were off;
// offWasOn = proportion of target-off units that were on.
// actThr, targThr are thresholds on the act and target values
// to count as active.
func (ly *Layer) TargetOnOff(di int, targVar, actVar string, actThr, targThr float32) (onWasOff, offWasOn float32) {
	tv := ly.unitVals(targVar, di)
	av := ly.unitVals(actVar, di)
	if tv == nil || av == nil {
		return math32.NaN(), math32.NaN()
	}
	trgOnSum, trgOffSum := float32(0), float32(0)
	for ui := range tv {
		if tv[ui] >= targThr {
			trgOnSum++
			if av[ui] < actThr {
				onWasOff++
			}
		} else {
			trgOffSum++
			if av[ui] >= actThr {
				offWasOn++
			}
		}
	}
	if trgOnSum > 0 {
		onWasOff /= trgOnSum
	}
	if trgOffSum > 0 {
		offWasOn /= trgOffSum
	}
	return
}

// ActAvgMax returns the average and maximum of the given unit variable
// (typically "Act") across all units in the layer,
// for data-parallel index di.
func (ly *Layer) ActAvgMax(di int, varNm string) (avg, max float32) {
	vals := ly.unitVals(varNm, di)
	if len(vals) == 0 {
		return math32.NaN(), math32.NaN()
	}
	max = vals[0]
	for _, v := range vals {
		avg += v
		if v > max {
			max = v
		}
	}
	avg /= float32(len(vals))
	return
}
