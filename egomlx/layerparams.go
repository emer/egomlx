// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package egomlx

import (
	"reflect"

	"cogentcore.org/core/base/reflectx"
	"github.com/emer/emergent/v2/params"
)

// LayerTypes enumerates all the different types of layers,
// for the different roles they play in a model.
// Class parameter styles automatically key off of these types.
type LayerTypes int32 //enums:enum

const (
	// The layer types.

	// Input is a layer that receives direct external input
	// in its Ext inputs.  It does not have any weight parameters:
	// its Act values are the applied external input patterns.
	Input LayerTypes = iota

	// Hidden is an internal layer that computes a nonlinear
	// transformation of the weighted inputs from its sending layers.
	Hidden

	// Target is a layer that receives direct external target inputs
	// used for computing error and driving learning.
	Target
)

// LayerParams contains all the parameters for a [Layer].
type LayerParams struct {
	// Layer is a back-reference to the layer these parameters belong to.
	Layer *Layer `display:"-"`

	// Act has the parameters for the activation function.
	Act ActParams `display:"inline"`
}

// Defaults sets default values for the LayerParams.
func (lp *LayerParams) Defaults() {
	lp.Act.Fn = ReLU
	lp.Act.LeakySlope = 0.01
}

// Update is called when a parameter is changed to update any
// dependent values.
func (lp *LayerParams) Update() {}

// StyleObject returns the object to apply styles to.
func (lp *LayerParams) StyleObject() any { return lp }

// StyleClass returns the space-separated list of class selectors (tags).
func (lp *LayerParams) StyleClass() string {
	cls := lp.Layer.Type.String()
	if lp.Layer.Class != "" {
		cls += " " + lp.Layer.Class
	}
	return cls
}

// StyleName returns the name of this object.
func (lp *LayerParams) StyleName() string { return lp.Layer.Name }

// ParamsString returns a listing of all parameters in the LayerParams.
// If nonDefault is true, only report those not at their default values.
func (lp *LayerParams) ParamsString(nonDefault bool) string {
	return params.PrintStruct(lp, 1, func(path string, ft reflect.StructField, fv any) bool {
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

var _ params.Styler = (*LayerParams)(nil)
