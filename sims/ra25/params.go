// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ra25

import (
	"github.com/emer/egomlx/egomlx"
)

// NetworkParams sets the minimal non-default global network params,
// controlling the loss function and optimizer for the backward pass.
// Base is always applied, and others can be optionally selected to apply on top of that.
var NetworkParams = egomlx.NetworkSheets{
	"Base": {
		{Sel: "Network", Doc: "standard backprop defaults: SSE loss, SGD optimizer",
			Set: func(np *egomlx.NetworkParams) {
				np.Loss = egomlx.SSE
				np.Optim = egomlx.SGD
				np.LRate = 0.5
				// clip update steps for stability of SGD with
				// saturating sigmoid outputs:
				np.ClipStepByValue = 0.1
			}},
	},
}

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
		{Sel: "Path", Doc: "all forward paths: bias on, Glorot weight init, learning on",
			Set: func(pt *egomlx.PathParams) {
				pt.UseBias = true
				// Glorot init keeps the sigmoid Output layer out of
				// saturation, enabling SGD to learn the mapping:
				pt.WtInit.Init = egomlx.Glorot
				pt.WtInit.Scale = 1
				pt.Learn = true
			}},
	},
}
