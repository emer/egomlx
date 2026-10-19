// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package egomlx

import (
	"reflect"

	"cogentcore.org/core/base/reflectx"
	"github.com/emer/emergent/v2/params"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/train/optimizer"
)

// LossFuns enumerates the loss functions that can be used to compute
// the error between the Act values of [Target] layers and their Targ
// values, which drives the backward learning pass.
type LossFuns int32 //enums:enum

const (
	// SSE is the sum of squared errors: 0.5 * sum over units of
	// (Act - Targ)^2, summed over target layers, and divided by NData
	// so the gradients are independent of the number of parallel data
	// items. This is the standard emergent / leabra error function for
	// backprop models, and it gives dLoss/dAct = (Act - Targ) / NData.
	SSE LossFuns = iota

	// MSE is the mean squared error: mean of (Act - Targ)^2 over all
	// units and data items, summed over target layers.
	MSE

	// BCE is the binary cross-entropy loss, summed over target layers:
	// -Targ * log(Act) - (1 - Targ) * log(1 - Act), averaged over all
	// units and data items. Requires a bounded 0-1 activation such as
	// [Sigmoid] on the target layers for the loss to be well-defined.
	BCE
)

// Optimizers enumerates the [gomlx] optimizers that drive the weight
// updates during the [Network] Backward pass.
//
//	optimizer = github.com/gomlx/gomlx/ml/train/optimizer
type Optimizers int32 //enums:enum

const (
	// SGD is plain stochastic gradient descent: Wt -= LRate * DWt.
	// See [NetworkParams.SGDDecay] for optional learning rate decay.
	SGD Optimizers = iota

	// Adam is the Adam optimizer (Kingma & Ba 2014), with adaptive
	// per-weight learning rates based on first and second moment
	// estimates of the gradients: [NetworkParams.Beta1], Beta2, Epsilon.
	Adam

	// AdamW is Adam with decoupled weight decay (AdamW), using
	// [NetworkParams.WeightDecay].
	AdamW

	// Adamax is Adam with an L-infinity (max) norm for the second
	// moment estimate, which can be better for gradients with spikes.
	Adamax

	// RMSProp divides the learning rate by a running average of the
	// recent gradient magnitudes (Adam without the first moment).
	RMSProp
)

// NetworkParams contains all the parameters for a [Network],
// which are global across the entire model, in contrast to the
// [LayerParams] and [PathParams] which apply at finer granularity.
//
// The optimizer parameters correspond to those used by the [gomlx]
// optimizer implementations, and those that are global there (e.g.,
// the learning rate -- [gomlx] optimizers do not support per-pathway
// rates) are global here as well. Anything that is pathway-specific
// (currently: whether the pathway learns at all, [PathParams.Learn])
// lives in the [PathParams].
//
//	optimizer = github.com/gomlx/gomlx/ml/train/optimizer
type NetworkParams struct {
	// Network is a back-reference to the network these parameters belong to.
	Network *Network `display:"-"`

	// Loss is the loss function computing the error signal driving
	// the backward pass, from Act vs. Targ values of Target layers.
	Loss LossFuns `default:"SSE"`

	// Optim is the optimizer used to update weights during Backward.
	Optim Optimizers `default:"SGD"`

	// LRate is the learning rate used by the optimizer.
	// If 0, the optimizer's own default applies: 0.1 for SGD,
	// 0.001 for the Adam family.
	LRate float64 `default:"0.1"`

	// SGDDecay enables learning rate decay for SGD, which divides
	// the learning rate by sqrt(global_step) over training.
	// Off by default, for a constant learning rate, following
	// the standard emergent backprop usage.
	SGDDecay bool

	// Beta1 is the Adam moving average coefficient for the gradient
	// (momentum) numerator. Requires [Adam] family optimizer.
	Beta1 float64 `default:"0.9"`

	// Beta2 is the Adam moving average coefficient for the variance
	// (second moment) denominator. Requires [Adam] family optimizer.
	Beta2 float64 `default:"0.999"`

	// Epsilon is the small constant added to the Adam denominator
	// for numerical stability. Default matches gomlx Adam: 1e-7.
	Epsilon float64 `default:"1e-7"`

	// WeightDecay is the decoupled weight decay strength for [AdamW],
	// which shrinks weights by LRate * WeightDecay each step.
	// 0 = no decay.
	WeightDecay float64 `default:"0"`

	// ClipStepByValue clips each weight update step (after multiplying
	// by the learning rate and applying the optimizer) to this
	// absolute value: +-ClipStepByValue. 0 = no clipping.
	ClipStepByValue float64 `default:"0"`

	// ClipNaN drops weight updates (keeping the prior weight values)
	// anywhere the update is NaN or +/-Inf, which can keep training
	// alive through transient instabilities.
	ClipNaN bool `default:"false"`
}

// Defaults sets default values for the NetworkParams.
func (np *NetworkParams) Defaults() {
	np.Loss = SSE
	np.Optim = SGD
	np.LRate = 0.1
	np.SGDDecay = false
	np.Beta1 = 0.9
	np.Beta2 = 0.999
	np.Epsilon = 1e-7
	np.WeightDecay = 0
	np.ClipStepByValue = 0
	np.ClipNaN = false
	np.Update()
}

// Update is called when a parameter is changed to update any
// dependent values.
func (np *NetworkParams) Update() {}

// ShouldDisplay returns false for parameters that don't apply
// to the current optimizer selection.
func (np *NetworkParams) ShouldDisplay(field string) bool {
	switch field {
	case "SGDDecay":
		return np.Optim == SGD
	case "Beta1", "Beta2", "Epsilon":
		return np.Optim == Adam || np.Optim == AdamW || np.Optim == Adamax
	case "WeightDecay":
		return np.Optim == AdamW
	default:
		return true
	}
}

// Optimizer returns the [gomlx] optimizer.Interface for the current
// parameter settings, which drives the weight updates during Backward.
//
//	optimizer = github.com/gomlx/gomlx/ml/train/optimizer
func (np *NetworkParams) Optimizer() optimizer.Interface {
	switch np.Optim {
	case Adam:
		ac := optimizer.Adam().Betas(np.Beta1, np.Beta2).Epsilon(np.Epsilon)
		if np.LRate > 0 {
			ac.LearningRate(np.LRate)
		}
		return ac.Done()
	case AdamW:
		ac := optimizer.Adam().Betas(np.Beta1, np.Beta2).Epsilon(np.Epsilon).WeightDecay(np.WeightDecay)
		if np.LRate > 0 {
			ac.LearningRate(np.LRate)
		}
		return ac.Done()
	case Adamax:
		ac := optimizer.Adam().Betas(np.Beta1, np.Beta2).Epsilon(np.Epsilon).Adamax()
		if np.LRate > 0 {
			ac.LearningRate(np.LRate)
		}
		return ac.Done()
	case RMSProp:
		rc := optimizer.RMSProp().Epsilon(np.Epsilon)
		if np.LRate > 0 {
			rc.LearningRate(np.LRate)
		}
		return rc.Done()
	default: // SGD
		sgd := optimizer.StochasticGradientDescent().WithDecay(np.SGDDecay)
		if np.LRate > 0 {
			sgd.WithLearningRate(np.LRate)
		}
		return sgd.Done()
	}
}

// SetStoreParams applies the parameters that the [gomlx] optimizers
// read from the [model] [model.Store] hyperparameters, at the root
// scope of the given store. This must be re-called before rebuilding
// the computation graphs for changes to take effect.
//
// The learning rate is maintained as a variable in the store by the
// [gomlx] optimizers, so if it has already been created by a prior
// graph build, its value is explicitly updated here.
//
//	optimizer = github.com/gomlx/gomlx/ml/train/optimizer
func (np *NetworkParams) SetStoreParams(store *model.Store) {
	root := store.RootScope()
	root.SetParam(optimizer.ParamClipStepByValue, np.ClipStepByValue)
	root.SetParam(optimizer.ParamClipNaN, np.ClipNaN)
	if np.LRate > 0 {
		lrVar := optimizer.LearningRateVar(root, dtypes.Float32, np.LRate)
		lrVar.MustSetValue(tensors.FromScalar(float32(np.LRate)))
	}
}

// StyleObject returns the object to apply styles to.
func (np *NetworkParams) StyleObject() any { return np }

// StyleClass returns the space-separated list of class selectors (tags).
// For the network-level params, the class is just "Network", and there
// are typically no additional classes.
func (np *NetworkParams) StyleClass() string { return "Network" }

// StyleName returns the name of this object.
func (np *NetworkParams) StyleName() string {
	if np.Network == nil {
		return "Network"
	}
	return np.Network.Name
}

// ParamsString returns a listing of all parameters in the NetworkParams.
// If nonDefault is true, only report those not at their default values.
func (np *NetworkParams) ParamsString(nonDefault bool) string {
	return params.PrintStruct(np, 1, func(path string, ft reflect.StructField, fv any) bool {
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

var _ params.Styler = (*NetworkParams)(nil)
