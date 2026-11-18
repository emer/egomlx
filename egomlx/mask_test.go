// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package egomlx

import (
	"math/rand/v2"
	"testing"

	"cogentcore.org/lab/tensor"
	"github.com/emer/emergent/v2/paths"
)

// TestMask checks that a PoolOneToOne path only connects the matching
// pools: the Mask has the expected connections, the unconnected weights
// are zero after initialization and stay zero with learning, and input to
// one pool does not drive the units in the other pool. A Full path has
// no Mask.
func TestMask(t *testing.T) {
	net := NewNetwork("TestMask")
	nData := 2
	net.SetMaxData(nData)
	inp := net.AddLayer4D("Input", Input, 1, 2, 3, 3)
	hid := net.AddLayer4D("Hidden", Hidden, 1, 2, 4, 4)
	out := net.AddLayer2D("Output", Target, 3, 3)
	p1 := net.ConnectLayers(inp, hid, paths.NewPoolOneToOne(), Forward)
	lat := net.ConnectLayers(hid, hid, paths.NewPoolOneToOne(), Lateral)
	pf := net.ConnectLayers(hid, out, paths.NewFull(), Forward)
	net.Defaults()
	net.Params.NCycles = 3
	out.Params.Act.Fn = Sigmoid
	if err := net.Build(); err != nil {
		t.Fatalf("Build: %v", err)
	}
	net.InitWeights()
	if pf.Mask != nil || pf.FanIn != 32 {
		t.Errorf("Full path: Mask %v, FanIn %d", pf.Mask != nil, pf.FanIn)
	}
	if p1.Mask == nil || p1.FanIn != 9 || lat.Mask == nil || lat.FanIn != 16 {
		t.Fatalf("PoolOneToOne: FanIn %d, lateral FanIn %d", p1.FanIn, lat.FanIn)
	}
	nRecv := hid.Shape.Len()
	connected := func(si, ri int) bool { return si/9 == ri/16 } // pool of each unit
	checkWts := func(when string) {
		for si := range inp.Shape.Len() {
			for ri := range nRecv {
				w := p1.KernelVals[si*nRecv+ri]
				if connected(si, ri) != (p1.Mask[si*nRecv+ri] == 1) {
					t.Fatalf("%s: mask wrong at send %d recv %d", when, si, ri)
				}
				if !connected(si, ri) && w != 0 {
					t.Fatalf("%s: unconnected weight send %d recv %d = %g", when, si, ri, w)
				}
			}
		}
	}
	checkWts("init")

	// input only in pool 0: the units in pool 1 get no input
	net.InitExt()
	for di := range nData {
		in := tensor.New[float32](inp.Shape.Len())
		for i := range 9 {
			in.SetFloat1D(1, i)
		}
		net.LayerByName("Input").ApplyExt(di, in)
	}
	net.ApplyExts()
	if err := net.Forward(); err != nil {
		t.Fatal(err)
	}
	net0 := 0.0
	for ui := 16; ui < 32; ui++ {
		if v := hid.UnitValue1D(varIndexDef("Net"), ui, 0); v != 0 {
			t.Errorf("pool 1 unit %d Net = %g with no input to pool 1", ui, v)
		}
		net0 += float64(hid.UnitValue1D(varIndexDef("Net"), ui-16, 0))
	}
	if net0 == 0 {
		t.Errorf("pool 0 has no net input")
	}

	// learning keeps the unconnected weights at zero
	rnd := rand.New(rand.NewPCG(1, 2))
	w0 := append([]float32{}, p1.KernelVals...)
	for range 50 {
		net.InitExt()
		for di := range nData {
			in := tensor.New[float32](inp.Shape.Len())
			tg := tensor.New[float32](out.Shape.Len())
			for i := range in.Len() {
				if rnd.Float32() < 0.3 {
					in.SetFloat1D(1, i)
				}
			}
			tg.SetFloat1D(1, rnd.IntN(tg.Len()))
			net.LayerByName("Input").ApplyExt(di, in)
			net.LayerByName("Output").ApplyExt(di, tg)
		}
		net.ApplyExts()
		if err := net.Backward(); err != nil {
			t.Fatal(err)
		}
	}
	p1.ReadbackWeights()
	checkWts("trained")
	changed := 0
	for i, w := range p1.KernelVals {
		if w != w0[i] {
			changed++
		}
	}
	if changed == 0 {
		t.Errorf("connected weights did not learn")
	}
}
