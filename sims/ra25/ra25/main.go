// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/emer/egomlx/sims/ra25"
	"github.com/emer/emergent/v2/egui"
)

// main runs the ra25 sim.
func main() { egui.Run[ra25.Sim, ra25.Config]() }
