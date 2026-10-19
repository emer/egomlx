// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package egomlx

import (
	"fmt"
)

// NeuronVars is the list of unit variable names available on
// the units in a network.
// This list determines what is shown in the NetView
// (and the order of vars list).
// Not all layers support all variables:
// input layers support Act, Ext (and Net);
// hidden and target layers support Act, Net, and Bias;
// target layers also support Targ.
// Unsupported variables safely return [math32.NaN].
var NeuronVars = []string{"Act", "Net", "Ext", "Targ", "Bias"}

// NeuronVarN is the number of unit variables.
var NeuronVarN = len(NeuronVars)

// NeuronVarIndex is a map of unit variable name to index
// in the NeuronVars list.
var NeuronVarIndex = func() map[string]int {
	m := make(map[string]int, NeuronVarN)
	for i, nm := range NeuronVars {
		m[nm] = i
	}
	return m
}()

// NeuronVarIndexByName returns the index of given variable name
// in the NeuronVars list, or -1 and error if not found.
func NeuronVarIndexByName(varNm string) (int, error) {
	i, ok := NeuronVarIndex[varNm]
	if !ok {
		return -1, fmt.Errorf("NeuronVarIndexByName: variable not found: %s", varNm)
	}
	return i, nil
}

// NeuronVarProps is the list of unit variable properties,
// with the key being the name of the variable,
// and the value giving a space-separated list of
// go-tag-style properties for that variable.
var NeuronVarProps = map[string]string{
	"Act":  `range:"1" desc:"activation of the layer" cat:"Unit"`,
	"Net":  `range:"1" desc:"net input, pre-activation, of the layer" cat:"Unit"`,
	"Ext":  `range:"1" desc:"external input, for input layers" cat:"Unit"`,
	"Targ": `range:"1" desc:"target input, for target layers" cat:"Unit"`,
	"Bias": `range:"0.5" desc:"dense layer bias, per receiving unit" cat:"Unit"`,
}

// SynapseVars is the list of synapse variable names available on
// the synapses in a network.
var SynapseVars = []string{"Wt"}

// SynapseVarN is the number of synapse variables.
var SynapseVarN = len(SynapseVars)

// SynapseVarIndex is a map of synapse variable name to index
// in the SynapseVars list.
var SynapseVarIndex = func() map[string]int {
	m := make(map[string]int, SynapseVarN)
	for i, nm := range SynapseVars {
		m[nm] = i
	}
	return m
}()

// SynapseVarIndexByName returns the index of given variable name
// in the SynapseVars list, or -1 and error if not found.
func SynapseVarIndexByName(varNm string) (int, error) {
	i, ok := SynapseVarIndex[varNm]
	if !ok {
		return -1, fmt.Errorf("SynapseVarIndexByName: variable not found: %s", varNm)
	}
	return i, nil
}

// SynapseVarProps is the list of synapse variable properties,
// with the key being the name of the variable,
// and the value giving a space-separated list of
// go-tag-style properties for that variable.
var SynapseVarProps = map[string]string{
	"Wt": `range:"0.5" desc:"weight of the connection" cat:"Wt"`,
}
