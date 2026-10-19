// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package egomlx provides an implementation of the [emer] neural network
// interfaces, built on the [gomlx] machine learning library, for
// constructing and visualizing neural network models using the
// [netview] network viewer.
//
//	emer = github.com/emer/emergent/v2/emer
//	gomlx = github.com/gomlx/gomlx
//	netview = github.com/emer/emergent/v2/netview
//
// The [Network], [Layer] and [Path] types implement the
// [emer.Network], [emer.Layer] and [emer.Path] interfaces,
// which support visualization in the [netview], parameter setting,
// and weights I/O, in exactly the same way as the [axon] and [leabra]
// packages do for their respective simulation frameworks.
//
//	axon = github.com/emer/axon/v2/axon
//	leabra = github.com/emer/leabra/v2/leabra
//
// Unlike those, which are full algorithmic simulation frameworks with
// their own computational implementations, egomlx is a thin toolkit
// for expressing standard machine learning models -- built with the
// [gomlx] API, which is all in Go, and includes a range of backends
// from pure Go CPU to Metal GPU and others -- in the emergent
// visualization system.
//
// # Data Organization
//
// The outer-most dimension of all network state (activation values and
// weights readback buffers) is the number of data items being processed
// in parallel (MaxData / NData, indexed by di), consistent with the
// axon convention, so that multiple input patterns can be batched and
// displayed together. Activation buffers have shape
// [NData][NUnits][NVars], with the variable index inner-most.
//
// # Forward Pass and Training
//
// The [Network.Forward] method runs a real forward pass through the
// [gomlx] model, reading back all activations and weights, so that the
// [netview] shows genuinely computed values.
//
// The [Network.Backward] method runs the backward learning pass: it
// computes the loss between the Act and Targ values of the [Target]
// layers, backpropagates gradients via [gomlx] automatic differentiation,
// and applies weight updates using the optimizer configured in the
// [NetworkParams], all within a single compiled computation graph.
// [Network.TrainTrial] runs Forward followed by Backward.
// The global loss function and optimizer parameters live in
// [NetworkParams], while pathway-specific learning parameters
// (currently [PathParams.Learn]) live in the [PathParams].
package egomlx

//go:generate core generate -add-types -add-funcs
