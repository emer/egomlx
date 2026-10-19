// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package egomlx

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/base/iox/tomlx"
	"cogentcore.org/core/core"
	"github.com/emer/emergent/v2/emer"
)

// SaveParamsSnapshot saves a snapshot of the current parameters
// to either `params_good` if good = true (for current good reference params)
// or `params_YYYY_MM_DD` datestamp directory,
// providing a snapshot of the simulation params for easy diffs and later reference.
// It also saves the given config (via TOML) alongside the params listings.
func (nt *Network) SaveParamsSnapshot(cfg any, good bool) error {
	date := time.Now().Format("2006_01_02")
	if good {
		date = "good"
	}
	dir := "params_" + date
	err := os.Mkdir(dir, 0775)
	if err != nil {
		log.Println(err) // notify but OK if it exists
	}
	fmt.Println("Saving params to:", dir)
	tomlx.Save(cfg, filepath.Join(dir, "config.toml"))
	nt.SaveParams(emer.AllParams, core.Filename(filepath.Join(dir, "params_all.txt")))
	nt.SaveParams(emer.NonDefault, core.Filename(filepath.Join(dir, "params_nondef.txt")))
	return nil
}

// LayerByName returns a layer by looking it up by name in the layer map
// (nil if not found).
func (nt *Network) LayerByName(name string) *Layer {
	ely, err := nt.EmerLayerByName(name)
	if err != nil {
		errors.Log(err)
		return nil
	}
	return ely.(*Layer)
}

// UpdateLayerMaps updates the [emer] [emer.NetworkBase] LayerNameMap
// and the [LayerClassMap].
func (nt *Network) UpdateLayerMaps() {
	nt.UpdateLayerNameMap()
	nt.LayerClassMap = make(map[string][]string)
	for _, ly := range nt.Layers {
		cs := ly.Type.String() + " " + ly.Class
		cls := strings.Split(cs, " ")
		for _, cl := range cls {
			if cl == "" {
				continue
			}
			ll := nt.LayerClassMap[cl]
			ll = append(ll, ly.Name)
			nt.LayerClassMap[cl] = ll
		}
	}
}

// LayersByType returns a list of layer names by given layer type(s).
func (nt *Network) LayersByType(layType ...LayerTypes) []string {
	var nms []string
	for _, tp := range layType {
		nm := tp.String()
		nms = append(nms, nm)
	}
	return nt.LayersByClass(nms...)
}

// LayersByClass returns a list of layer names by given class(es).
// If no classes are passed, all layer names in order are returned.
// Layers that are Off are not included.
func (nt *Network) LayersByClass(classes ...string) []string {
	if nt.LayerClassMap == nil {
		nt.UpdateLayerMaps()
	}
	var nms []string
	if len(classes) == 0 {
		for _, ly := range nt.Layers {
			if ly.Off {
				continue
			}
			nms = append(nms, ly.Name)
		}
		return nms
	}
	for _, lc := range classes {
		nms = append(nms, nt.LayerClassMap[lc]...)
	}
	return nms
}

// AddLayerClass adds a CSS-style class name(s) for the given layer,
// which is then used for class-based parameter styling.
func (ly *Layer) AddLayerClass(cls ...string) *Layer {
	ly.AddClass(cls...)
	return ly
}

// String returns a string representation of the network,
// with a summary of the layers and paths.
func (nt *Network) String() string {
	var s strings.Builder
	s.WriteString("Network: " + nt.Name + "\n")
	for _, ly := range nt.Layers {
		s.WriteString(fmt.Sprintf("  %s: %s %s", ly.Name, ly.Type.String(), ly.Shape.String()))
		for _, pt := range ly.RecvPaths {
			s.WriteString(fmt.Sprintf(" <- %s", pt.Send.Name))
		}
		s.WriteString("\n")
	}
	return s.String()
}
