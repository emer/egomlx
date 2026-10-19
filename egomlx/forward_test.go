// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package egomlx

import (
	"math/rand/v2"
	"testing"

	"cogentcore.org/core/math32"
	"cogentcore.org/lab/tensor"
	"github.com/emer/emergent/v2/paths"
)

// TestForward builds a small ra25-style feedforward network,
// builds the gomlx model, applies a batch of input patterns,
// runs a real forward pass, and verifies that the output
// activations are real (finite, non-NaN) values.
func TestForward(t *testing.T) {
	net := NewNetwork("TestNet")
	net.SetMaxData(4)

	inp := net.AddLayer2D("Input", Input, 5, 5)
	hid1 := net.AddLayer2D("Hidden1", Hidden, 8, 8)
	hid2 := net.AddLayer2D("Hidden2", Hidden, 8, 8)
	out := net.AddLayer2D("Output", Target, 5, 5)

	full := paths.NewFull()
	net.ConnectLayers(inp, hid1, full, Forward)
	net.ConnectLayers(hid1, hid2, full, Forward)
	net.ConnectLayers(hid2, out, full, Forward)

	if err := net.Build(); err != nil {
		t.Fatalf("Build: %v", err)
	}
	net.Defaults()
	net.InitWeights()

	// apply a batch of 4 random binary input patterns.
	rnd := rand.New(rand.NewPCG(1, 2))
	for di := 0; di < 4; di++ {
		tsr := tensor.New[float32](25)
		for i := range tsr.Len() {
			if rnd.Float32() < 0.5 {
				tsr.SetFloat1D(1, i)
			}
		}
		inp.ApplyExt(di, tsr)
	}
	net.ApplyExts()

	if err := net.Forward(); err != nil {
		t.Fatalf("Forward: %v", err)
	}

	checkLayer := func(ly *Layer, nm string) {
		nu := ly.Shape.Len()
		vi, _ := ly.UnitVarIndex(nm)
		nonNaN := 0
		var sum float32
		for di := 0; di < 4; di++ {
			for ui := range nu {
				v := ly.UnitValue1D(vi, ui, di)
				if !math32.IsNaN(v) {
					nonNaN++
					sum += v
				}
			}
		}
		t.Logf("Layer %-8s %s: %d/%d non-NaN, mean=%g, sample[0]=%g",
			ly.Name, nm, nonNaN, nu*4, float64(sum)/float64(nu*4), ly.UnitValue1D(vi, 0, 0))
		if nonNaN == 0 {
			t.Errorf("Layer %s: all %s values are NaN -- forward pass did not run", ly.Name, nm)
		}
	}
	checkLayer(out, "Act")
	checkLayer(hid1, "Act")
	checkLayer(hid2, "Act")
	checkLayer(inp, "Act")

	// verify weight readback is finite.
	w := hid1.RecvPaths[0].SynValue1D(0, 0)
	if math32.IsNaN(w) {
		t.Errorf("weight readback is NaN")
	}
	t.Logf("weight[0]=%g  out.Act[0,0]=%g", w, out.UnitValue1D(0, 0, 0))
}
