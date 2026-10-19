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

// testRANet builds a small ra25-style feedforward network with a
// 5x5 Input, two 8x8 Hidden layers, and a 5x5 Target Output layer,
// fully connected, built and with weights initialized.
func testRANet(t *testing.T, name string, nData int) *Network {
	t.Helper()
	net := NewNetwork(name)
	net.SetMaxData(nData)

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
	return net
}

// applyRandomBatch applies nData random binary input patterns and
// random binary target patterns to the network, for the given di
// items, and applies the external inputs to start a trial.
func applyRandomBatch(t *testing.T, net *Network, rnd *rand.Rand, nData int) {
	t.Helper()
	nu := 25
	net.InitExt()
	for di := 0; di < nData; di++ {
		inTsr := tensor.New[float32](nu)
		tgTsr := tensor.New[float32](nu)
		for i := range nu {
			if rnd.Float32() < 0.5 {
				inTsr.SetFloat1D(1, i)
			}
			if rnd.Float32() < 0.5 {
				tgTsr.SetFloat1D(1, i)
			}
		}
		net.LayerByName("Input").ApplyExt(di, inTsr)
		net.LayerByName("Output").ApplyExt(di, tgTsr)
	}
	net.ApplyExts()
}

// sumAbs returns the sum of absolute values of a float32 slice.
func sumAbs(v []float32) float64 {
	var s float64
	for _, x := range v {
		s += float64(math32.Abs(x))
	}
	return s
}

// TestBackward tests the full backward learning pass: the loss
// should decrease substantially over repeated training trials on
// the same input / target patterns, DWt gradients should be non-zero,
// weights should change, and the global step should increment.
func TestBackward(t *testing.T) {
	net := testRANet(t, "TestBwd", 4)
	nt := net
	nt.Params.Optim = SGD
	nt.Params.LRate = 0.5
	nt.Params.ClipStepByValue = 0.1
	// Glorot init keeps the sigmoid output layer out of saturation,
	// which is the standard choice for ra-style backprop models:
	for _, ly := range net.Layers {
		for _, pt := range ly.RecvPaths {
			pt.Params.WtInit.Init = Glorot
		}
	}
	nt.UpdateParams()
	nt.InitWeights()

	rnd := rand.New(rand.NewPCG(3, 4))
	applyRandomBatch(t, net, rnd, 4)

	w0 := append([]float32{}, net.Layers[1].RecvPaths[0].KernelVals...)

	if err := net.Backward(); err != nil {
		t.Fatalf("Backward: %v", err)
	}
	firstLoss := net.Loss
	if math32.IsNaN(firstLoss) {
		t.Fatalf("Backward: initial loss is NaN")
	}
	t.Logf("initial loss: %g", firstLoss)

	const nTrials = 1000
	for i := 0; i < nTrials; i++ {
		if err := net.TrainTrial(DWtReadRaw); err != nil {
			t.Fatalf("TrainTrial %d: %v", i, err)
		}
		if math32.IsNaN(net.Loss) {
			t.Fatalf("TrainTrial %d: loss is NaN", i)
		}
		if i == 0 {
			// DWt values should have been read back for all paths:
			for _, ly := range net.Layers {
				for _, pt := range ly.RecvPaths {
					if sumAbs(pt.KernelDWt) <= 0 {
						t.Errorf("path %s: KernelDWt is all zero after Backward with DWtReadRaw", pt.Name)
					}
					if sumAbs(pt.BiasDWt) <= 0 {
						t.Errorf("path %s: BiasDWt is all zero after Backward with DWtReadRaw", pt.Name)
					}
				}
			}
		}
	}
	finalLoss := net.Loss
	t.Logf("final loss after %d trials: %g (first: %g)", nTrials, finalLoss, firstLoss)
	if finalLoss >= firstLoss*0.5 {
		t.Errorf("loss did not decrease sufficiently: first %g -> final %g", firstLoss, finalLoss)
	}

	// weights should have changed:
	w1 := net.Layers[1].RecvPaths[0].KernelVals
	changed := 0
	for i := range w0 {
		if w0[i] != w1[i] {
			changed++
		}
	}
	if changed == 0 {
		t.Errorf("no weights changed during training")
	}
	t.Logf("weights changed: %d / %d", changed, len(w0))

	if net.GlobalStep() < nTrials {
		t.Errorf("GlobalStep = %d, want >= %d", net.GlobalStep(), nTrials)
	}
}

// TestBackwardAdam verifies that the Adam optimizer option also
// trains the network, decreasing the loss.
func TestBackwardAdam(t *testing.T) {
	net := testRANet(t, "TestBwdAdam", 4)
	// Adam with its default lr (0.001) is slow: use a faster one:
	net.Params.Optim = Adam
	net.Params.LRate = 0.01
	net.UpdateParams()

	rnd := rand.New(rand.NewPCG(5, 6))
	applyRandomBatch(t, net, rnd, 4)

	if err := net.Backward(); err != nil {
		t.Fatalf("Backward: %v", err)
	}
	firstLoss := net.Loss
	if math32.IsNaN(firstLoss) {
		t.Fatalf("Backward: initial loss is NaN")
	}
	const nTrials = 1000
	for i := 0; i < nTrials; i++ {
		if err := net.TrainTrial(); err != nil {
			t.Fatalf("TrainTrial %d: %v", i, err)
		}
	}
	t.Logf("Adam: initial loss %g -> final loss %g", firstLoss, net.Loss)
	if math32.IsNaN(net.Loss) || net.Loss >= firstLoss {
		t.Errorf("Adam did not decrease loss: first %g -> final %g", firstLoss, net.Loss)
	}
}

// TestBackwardLearnOff verifies that a pathway with Learn = false
// keeps its weights fixed and has no DWt values.
func TestBackwardLearnOff(t *testing.T) {
	net := testRANet(t, "TestBwdLearnOff", 4)
	net.Params.LRate = 0.5

	pt := net.LayerByName("Hidden1").RecvPaths[0]
	pt.Params.Learn = false
	net.UpdateParams()

	rnd := rand.New(rand.NewPCG(7, 8))
	applyRandomBatch(t, net, rnd, 4)

	w0 := append([]float32{}, pt.KernelVals...)
	for i := 0; i < 50; i++ {
		if err := net.TrainTrial(DWtReadRaw); err != nil {
			t.Fatalf("TrainTrial %d: %v", i, err)
		}
	}
	if sumAbs(pt.KernelDWt) != 0 || sumAbs(pt.BiasDWt) != 0 {
		t.Errorf("Learn=false path should have zero DWt")
	}
	for i := range w0 {
		if w0[i] != pt.KernelVals[i] {
			t.Errorf("Learn=false path weight changed: %d: %g -> %g", i, w0[i], pt.KernelVals[i])
			break
		}
	}
	// other paths should still be learning:
	pt2 := net.LayerByName("Output").RecvPaths[0]
	if sumAbs(pt2.KernelDWt) <= 0 {
		t.Errorf("learning path should have non-zero DWt")
	}
}

// TestBackwardNoTargets verifies that a network without any Target
// layers returns an error on Backward.
func TestBackwardNoTargets(t *testing.T) {
	net := NewNetwork("TestBwdNoTarg")
	net.SetMaxData(1)
	inp := net.AddLayer2D("Input", Input, 5, 5)
	hid := net.AddLayer2D("Hidden", Hidden, 4, 4)
	full := paths.NewFull()
	net.ConnectLayers(inp, hid, full, Forward)
	if err := net.Build(); err != nil {
		t.Fatalf("Build: %v", err)
	}
	net.Defaults()
	net.InitWeights()
	if err := net.Backward(); err == nil {
		t.Errorf("Backward should return error for network with no Target layers")
	} else {
		t.Logf("expected error: %v", err)
	}
}

// TestBackwardNoTargs verifies that Backward returns an error when
// the target values have not been applied (all NaN).
func TestBackwardNoTargs(t *testing.T) {
	net := testRANet(t, "TestBwdNoTargs", 2)
	rnd := rand.New(rand.NewPCG(9, 10))
	net.InitExt()
	tsr := tensor.New[float32](25)
	for i := range 25 {
		tsr.SetFloat1D(rnd.Float64(), i)
	}
	net.LayerByName("Input").ApplyExt(0, tsr)
	net.ApplyExts() // no target values applied
	if err := net.Backward(); err == nil {
		t.Errorf("Backward should return error when targets are not applied")
	} else {
		t.Logf("expected error: %v", err)
	}
}
