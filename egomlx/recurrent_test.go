// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package egomlx

import (
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/emer/emergent/v2/paths"
)

// testRecurrentNet builds a recurrent network: Input -> Hidden <-> Output,
// with a Lateral path within Hidden, and the given number of cycles and
// Hidden noise.
func testRecurrentNet(t *testing.T, nData, ncyc int, noise float64) *Network {
	t.Helper()
	net := NewNetwork("TestRecurrent")
	net.SetMaxData(nData)
	inp := net.AddLayer2D("Input", Input, 5, 5)
	hid := net.AddLayer2D("Hidden", Hidden, 8, 8)
	out := net.AddLayer2D("Output", Target, 5, 5)
	full := paths.NewFull()
	net.ConnectLayers(inp, hid, full, Forward)
	net.ConnectLayers(hid, hid, full, Lateral)
	net.ConnectLayers(hid, out, full, Forward)
	net.ConnectLayers(out, hid, full, Back)
	net.Defaults()
	net.Params.NCycles = ncyc
	hid.Params.Act.Noise = noise
	out.Params.Act.Fn = Sigmoid
	if err := net.Build(); err != nil {
		t.Fatalf("Build: %v", err)
	}
	net.InitWeights()
	return net
}

func outActs(net *Network) []float32 {
	vals := []float32{}
	out := net.LayerByName("Output")
	for ui := range out.Shape.Len() {
		for di := range net.NData {
			vals = append(vals, out.UnitValue1D(varIndexDef("Act"), ui, di))
		}
	}
	return vals
}

// TestRecurrent checks that a network with Back and Lateral paths builds,
// that the Forward path order check exempts the recurrent paths, that
// more cycles of settling change the output (the recurrent paths are
// used), that noise makes repeated forward passes differ, and that it
// learns with backpropagation through time.
func TestRecurrent(t *testing.T) {
	nData := 4
	rnd := rand.New(rand.NewPCG(1, 2))

	// a back path into an earlier layer must not be a Forward path
	bad := NewNetwork("Bad")
	bad.SetMaxData(1)
	bi := bad.AddLayer2D("Input", Input, 2, 2)
	bh := bad.AddLayer2D("Hidden", Hidden, 2, 2)
	bo := bad.AddLayer2D("Output", Target, 2, 2)
	bad.ConnectLayers(bi, bh, paths.NewFull(), Forward)
	bad.ConnectLayers(bh, bo, paths.NewFull(), Forward)
	bad.ConnectLayers(bo, bh, paths.NewFull(), Forward)
	if err := bad.Build(); err == nil {
		t.Errorf("Forward path from a later layer should not build")
	}

	net1 := testRecurrentNet(t, nData, 1, 0)
	net5 := testRecurrentNet(t, nData, 5, 0)
	// same weights in both
	for li, ly := range net1.Layers {
		for pi, pt := range ly.RecvPaths {
			pt5 := net5.Layers[li].RecvPaths[pi]
			for si := range pt.Send.Shape.Len() {
				for ri := range pt.Recv.Shape.Len() {
					pt5.SetWtValue(si, ri, pt.SynValue1D(0, pt.SynIndex(si, ri)))
				}
			}
		}
	}
	applyRandomBatch(t, net1, rnd, nData)
	rnd = rand.New(rand.NewPCG(1, 2))
	applyRandomBatch(t, net5, rnd, nData)
	if err := net1.Forward(); err != nil {
		t.Fatal(err)
	}
	if err := net5.Forward(); err != nil {
		t.Fatal(err)
	}
	if slices.Equal(outActs(net1), outActs(net5)) {
		t.Errorf("5 cycles of settling gave the same output as 1 cycle")
	}

	noisy := testRecurrentNet(t, nData, 5, 0.5)
	applyRandomBatch(t, noisy, rnd, nData)
	noisy.Forward()
	a1 := outActs(noisy)
	noisy.Forward()
	if slices.Equal(a1, outActs(noisy)) {
		t.Errorf("noise did not change the output across forward passes")
	}

	// learning
	net5.Params.Optim = Adam
	net5.Params.LRate = 0.01
	net5.Params.Loss = BCE
	net5.UpdateParams()
	net5.Backward()
	first := net5.Loss
	for range 200 {
		net5.Backward()
	}
	t.Logf("recurrent loss: %.4f -> %.4f", first, net5.Loss)
	if !(net5.Loss < 0.5*first) {
		t.Errorf("recurrent network did not learn: loss %g -> %g", first, net5.Loss)
	}
}

// TestCarryover checks that with Decay < 1, the output for the same input
// depends on the previous trial (the activity carries over), and that it
// does not with Decay = 1, and that the network still learns.
func TestCarryover(t *testing.T) {
	nData := 2
	outAfter := func(net *Network, prevSeed, seed uint64) []float32 {
		net.InitActs()
		applyRandomBatch(t, net, rand.New(rand.NewPCG(prevSeed, 9)), nData)
		net.Forward()
		applyRandomBatch(t, net, rand.New(rand.NewPCG(seed, 9)), nData)
		net.Forward()
		return outActs(net)
	}
	for _, decay := range []float64{1, 0.5} {
		net := testRecurrentNet(t, nData, 5, 0)
		net.Params.Decay = decay
		net.UpdateParams()
		a := outAfter(net, 1, 3)
		b := outAfter(net, 2, 3)
		same := slices.Equal(a, b)
		t.Logf("decay %g: same output after a different previous trial: %v", decay, same)
		if decay == 1 && !same {
			t.Errorf("no carryover expected with Decay = 1")
		}
		if decay < 1 && same {
			t.Errorf("carryover expected with Decay = %g", decay)
		}
		if decay < 1 {
			net.Params.Optim = Adam
			net.Params.LRate = 0.01
			net.Params.Loss = BCE
			net.UpdateParams()
			rnd := rand.New(rand.NewPCG(5, 6))
			applyRandomBatch(t, net, rnd, nData)
			net.TrainTrial()
			first := net.Loss
			for range 200 {
				net.TrainTrial()
			}
			t.Logf("carryover loss: %.4f -> %.4f", first, net.Loss)
			if !(net.Loss < 0.5*first) {
				t.Errorf("carryover network did not learn: %g -> %g", first, net.Loss)
			}
		}
	}
}

// TestDropout checks that dropout does not affect Forward (testing)
// passes, that it does affect the training pass, and that the network
// still learns with it.
func TestDropout(t *testing.T) {
	nData := 4
	net := testRANet(t, "TestDropout", nData)
	net.Params.Optim = Adam
	net.Params.LRate = 0.01
	net.UpdateParams()
	applyRandomBatch(t, net, rand.New(rand.NewPCG(3, 4)), nData)
	net.Forward()
	a := outActs(net)
	for _, ly := range net.Layers {
		if ly.Type == Hidden {
			ly.Params.Act.Dropout = 0.5
		}
	}
	net.UpdateParams()
	net.Forward()
	if !slices.Equal(a, outActs(net)) {
		t.Errorf("dropout changed the Forward output")
	}
	// training loss varies across passes with the same weights (no
	// learning) because of the random dropout
	for _, pt := range net.LayerByName("Hidden1").RecvPaths {
		pt.Params.Learn = false
	}
	for _, pt := range net.LayerByName("Hidden2").RecvPaths {
		pt.Params.Learn = false
	}
	for _, pt := range net.LayerByName("Output").RecvPaths {
		pt.Params.Learn = false
	}
	net.UpdateParams()
	net.Backward()
	l1 := net.Loss
	net.Backward()
	if l1 == net.Loss {
		t.Errorf("dropout did not change the training pass: loss %g both times", l1)
	}
	for _, ly := range net.Layers {
		for _, pt := range ly.RecvPaths {
			pt.Params.Learn = true
		}
	}
	net.UpdateParams()
	net.Backward()
	first := net.Loss
	for range 300 {
		net.Backward()
	}
	t.Logf("dropout loss: %.4f -> %.4f", first, net.Loss)
	if !(net.Loss < 0.5*first) {
		t.Errorf("did not learn with dropout: %g -> %g", first, net.Loss)
	}
}
