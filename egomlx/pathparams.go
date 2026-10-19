// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package egomlx

import (
	"reflect"

	"cogentcore.org/core/base/reflectx"
	"github.com/emer/emergent/v2/params"
)

// PathTypes enumerates all the different types of pathways,
// for the different roles they play in a model.
// Class parameter styles automatically key off of these types.
type PathTypes int32 //enums:enum

const (
	// The path types.

	// Forward is a standard forward pathway from a sending layer
	// to a receiving layer, implemented as a dense (linear) transformation.
	// This is the only path type in the current feedforward-only toolkit.
	Forward PathTypes = iota

	// Back is a top-down back-going projection pathway,
	// for recurrent networks.
	Back
)

// WtInits enumerates the types of weight initialization
// distributions that can be applied to path weights.
type WtInits int32 //enums:enum

const (
	// He uses He (2015) initialization: a normal distribution with
	// standard deviation sqrt(1 / fanIn) -- designed for ReLU activations.
	He WtInits = iota

	// Glorot uses Glorot (Xavier) initialization: a normal distribution
	// with standard deviation sqrt(1 / (fanIn + fanOut)) -- designed
	// for sigmoid / tanh activations.
	Glorot

	// Normal uses a normal (Gaussian) distribution with
	// mean and standard deviation from the [WtInitParams] fields.
	Normal

	// TruncNormal uses a truncated normal distribution, with values
	// clipped to +- 2 standard deviations.
	TruncNormal

	// Uniform uses a uniform distribution between -limit and +limit,
	// where limit = sqrt(3 / (fanIn + fanOut)) * scale.
	Uniform

	// Zero initializes all weights to zero -- biases only typically,
	// so that the network starts in an identity-like state.
	Zero
)

// WtInitParams has the parameters for initializing the weight values
// of a pathway.
type WtInitParams struct {
	// Kind is the type of weight initialization to use.
	Init WtInits `default:"He"`

	// Mean is the mean of the distribution, for Normal and TruncNormal.
	Mean float64 `default:"0"`

	// Scale is the scale (standard deviation for Normal and TruncNormal,
	// or multiplier of the default limit) for the distribution.
	Scale float64 `default:"1"`
}

// PathParams contains all the parameters for a [Path].
type PathParams struct {
	// Path is a back-reference to the path these parameters belong to.
	Path *Path `display:"-"`

	// Type is the type of the pathway (Forward).
	// This drives the parameter style keying.
	Type PathTypes `display:"inline"`

	// UseBias indicates whether the dense transformation includes
	// a bias term.  Requires rebuilding the network graph when changed.
	UseBias bool `default:"true"`

	// WtInit has the parameters for weight initialization.
	WtInit WtInitParams `display:"inline"`

	// Learn indicates whether the weights of this pathway are trainable
	// (learnable) during the Backward pass. If false, the weights stay
	// fixed, acting as a constant transformation.
	// The global [NetworkParams.LRate] learning rate applies to all
	// pathways that are learning: [gomlx] optimizers do not support
	// per-pathway rates.
	Learn bool `default:"true"`
}

// Defaults sets default values for the PathParams.
func (pp *PathParams) Defaults() {
	pp.Type = Forward
	pp.UseBias = true
	pp.WtInit.Init = He
	pp.WtInit.Mean = 0
	pp.WtInit.Scale = 1
	pp.Learn = true
}

// Update is called when a parameter is changed to update any
// dependent values.
func (pp *PathParams) Update() {}

// StyleObject returns the object to apply styles to.
func (pp *PathParams) StyleObject() any { return pp }

// StyleClass returns the space-separated list of class selectors (tags).
func (pp *PathParams) StyleClass() string {
	cls := pp.Path.TypeName()
	if pp.Path.Class != "" {
		cls += " " + pp.Path.Class
	}
	return cls
}

// StyleName returns the name of this object.
func (pp *PathParams) StyleName() string { return pp.Path.Name }

// ParamsString returns a listing of all parameters in the PathParams.
// If nonDefault is true, only report those not at their default values.
func (pp *PathParams) ParamsString(nonDefault bool) string {
	return params.PrintStruct(pp, 1, func(path string, ft reflect.StructField, fv any) bool {
		if ft.Tag.Get("display") == "-" {
			return false
		}
		if nonDefault {
			if def := ft.Tag.Get("default"); def != "" {
				if reflectx.ValueIsDefault(reflect.ValueOf(fv), def) {
					return false
				}
			} else {
				if reflectx.NonPointerType(ft.Type).Kind() != reflect.Struct {
					return false
				}
			}
		}
		return true
	}, nil)
}

var _ params.Styler = (*PathParams)(nil)
