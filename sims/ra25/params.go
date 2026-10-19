// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ra25

import (
	"github.com/emer/egomlx/egomlx"
)

// LayerParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var LayerParams = egomlx.LayerSheets{
	"Base": {
		{Sel: "Layer", Doc: "all defaults",
			Set: func(ly *egomlx.LayerParams) {
				// all at defaults
			}},
		{Sel: ".Hidden", Doc: "hidden layers use ReLU activation",
			Set: func(ly *egomlx.LayerParams) {
				ly.Act.Fn = egomlx.ReLU
			}},
		{Sel: "#Output", Doc: "output layer uses a Sigmoid, giving bounded output activations",
			Set: func(ly *egomlx.LayerParams) {
				ly.Act.Fn = egomlx.Sigmoid
			}},
	},
}

// PathParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var PathParams = egomlx.PathSheets{
	"Base": {
		{Sel: "Path", Doc: "all forward paths: bias on, He weight init, standard backprop rate",
			Set: func(pt *egomlx.PathParams) {
				pt.UseBias = true
				pt.WtInit.Init = egomlx.He
				pt.WtInit.Scale = 1
				pt.LRate = 0.1
			}},
	},
}
