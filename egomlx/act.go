// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package egomlx

import (
	"github.com/gomlx/gomlx/ml/layers/activation"
)

// ActParams has the parameters for the activation function
// applied to the net input of a layer.
type ActParams struct {
	// Fn is the activation function to use.
	Fn ActFuns `default:"ReLU"`

	// LeakySlope is the slope for the LeakyReLU activation function.
	LeakySlope float64 `default:"0.01"`

	// Noise is the standard deviation of Gaussian noise added to the net
	// input of the layer on every cycle (see [NetworkParams.NCycles]),
	// in both training and testing. 0 = no noise.
	Noise float64 `default:"0"`

	// Dropout is the probability of setting each unit's activity to zero,
	// on each cycle of the training passes only (Backward), with the other
	// activities scaled by 1 / (1 - Dropout) to keep the mean the same.
	// It has no effect on Forward (testing) passes. 0 = no dropout.
	Dropout float64 `default:"0" min:"0" max:"1"`
}

// ActFuns enumerates the activation functions that can be applied
// to the net input of a layer, after its dense (linear) transformation.
// These map onto the [activation] function types in [gomlx].
//
//	activation = github.com/gomlx/gomlx/ml/layers/activation
//	gomlx = github.com/gomlx/gomlx
type ActFuns int32 //enums:enum

const (
	// The activation function types.

	// Linear is no activation function -- the identity, so that
	// the Act values are the same as the Net values.
	Linear ActFuns = iota

	// ReLU is the Rectified Linear Unit: max(0, x).
	ReLU

	// LeakyReLU is a Leaky Rectified Linear Unit: x for x > 0,
	// otherwise slope * x, where slope is given by [ActParams.LeakySlope].
	LeakyReLU

	// Sigmoid is the logistic sigmoid function: 1 / (1 + exp(-x)).
	Sigmoid

	// HardSigmoid is a piecewise linear approximation of the sigmoid.
	HardSigmoid

	// Swish is the Swish (SiLU) function: x * sigmoid(x).
	Swish

	// SELU is the Scaled Exponential Linear Unit.
	SELU

	// Tanh is the hyperbolic tangent function.
	Tanh

	// GELU is the Gaussian Error Linear Unit.
	GELU

	// GELUApprox is a computationally cheaper approximation of GELU.
	GELUApprox
)

// MLX returns the corresponding [activation.Type] for given ActFuns.
func (af ActFuns) MLX() activation.Type {
	switch af {
	case ReLU:
		return activation.TypeRelu
	case LeakyReLU:
		return activation.TypeLeakyRelu
	case Sigmoid:
		return activation.TypeSigmoid
	case HardSigmoid:
		return activation.TypeHardSigmoid
	case Swish:
		return activation.TypeSwish
	case SELU:
		return activation.TypeSelu
	case Tanh:
		return activation.TypeTanh
	case GELU:
		return activation.TypeGelu
	case GELUApprox:
		return activation.TypeGeluApprox
	case Linear:
		return activation.TypeNone
	}
	return activation.TypeNone
}
