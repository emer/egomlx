# EgoMLX Random Associator 25 Example

This example project serves as a demo and basic template for [EgoMLX](https://github.com/emer/egomlx) models -- you can use this as a starting point for creating your own models. It has been constructed to illustrate the most common types of functionality needed across models, to facilitate copy-paste programming.

It has a random-associator four-layer feedforward network -- built with the [gomlx](https://github.com/gomlx/gomlx) API and expressed through the [egomlx](https://github.com/emer/egomlx/egomlx) toolkit -- that learns mappings between 25 random input / output patterns defined over 5x5 input / output layers (25 units each):

* an `Input` 5x5 layer
* two `Hidden` 10x10 layers (sizes configurable)
* an `Output` 5x5 (target) layer

with full (dense) connections between successive layers. The data-parallel dimension (`NData`) is the outer-most dimension of all network state, per trial.

> **Note:** both the forward pass and the backward learning pass are fully implemented and live -- activations, the output error statistic, backpropagated gradients (DWt), and optimizer weight updates are all real, computed by the gomlx backend, so the network genuinely learns. The loss function (SSE by default) and optimizer (SGD by default, with Adam / AdamW / Adamax / RMSProp available) are configured in the `NetworkParams` sheet in `params.go`, and DWt values can be viewed in the netview when inspecting synapses. This example is also a template for constructing, visualizing, and running gomlx models in the emergent [netview](https://github.com/emer/emergent/wiki/NetView).

# Running the model

EgoMLX and emergent use the [Cogent Core](https://cogentcore.org/core) GUI framework. See [install](https://www.cogentcore.org/core/setup/install) instructions there. Once those prerequisites are in place, then the simplest way to run a simulation is:

    $ core run [platform]

run from this directory, where `[platform]` is optional (defaults to your local system) and can include `android`, `ios` and `web`!

Or directly with go:

    $ go run ./ra25
    $ go run ./ra25 --gui=false   # headless, writes log files

## Basic running and graphing

The basic interface has a toolbar across the top, a set of `Sim` fields on the left, and a tab bar with the network view (and other windows) on the right. The toolbar actions all have tooltips so just hover over those to see what they do.

* `Init` re-initializes the run with the same random seed (the `New Seed` toolbar item generates a new one).
* `Train` runs the training loops; `Stop` and `Step` control the run at the trial level.
* The `Network` tab shows the [netview](https://github.com/emer/emergent/wiki/NetView) visualization of the model: the four layers, their units, and the dense pathways. The bottom of the view shows the current loop counters, data-parallel index, trial name, and the live output error.
* You can click on units, pathways, and layers to inspect and edit their values and parameters, and to view them in the rasters and 3D scene.
* To speed up processing, turn off the `ViewOn` flag on the `TrainUpdate` / `TestUpdate` fields while running.

## Data parallelism

`Run.NData` sets how many input patterns are processed in parallel per trial (an effective mini-batch). The netview `Di` control at the bottom selects which data item is shown.

## Running from command line

    ./ra25 -help

To see a list of args that you can pass -- passing any arg will cause the model to run without the GUI, and write log files (see `config.toml` `[Log]` settings for which levels) into the `Root/` directory, plus optionally final weights files for each run.

# Code organization and notes

Most of the code is commented and should be read directly for how to do things. Here are just a few general organizational notes about code structure overall:

* The `egomlx` package (`github.com/emer/egomlx/egomlx`) implements the [emer](https://github.com/emer/emergent/wiki/Emer) `Network`, `Layer`, and `Path` interfaces on top of [gomlx](https://github.com/gomlx/gomlx), providing the standard emergent functionality: netview visualization, parameter setting via [params](https://github.com/emer/emergent/wiki/Params), and weights I/O.
* This sim mirrors the structure of [axon/v2/sims/ra25](https://github.com/emer/axon/tree/main/v2/sims/ra25): a `Sim` type with `ConfigSim`, `Init`, `ConfigLoops`, `ConfigStats`, `ConfigGUI`, and `RunNoGUI` methods.
* In Go, you can organize things however you want -- there are no constraints on order in Go code.
* The GUI config and elements are all optional and the `--gui=false` startup arg, along with other args, allows the model to be run without the GUI.
* The params editor can easily save to a file -- you can switch your project to using that as its default set of params to then easily always be using whatever params were saved last.
* The 25 random patterns are in the embedded `random_5x5_25.tsv` file; `ConfigInputs` shows how to regenerate them.
