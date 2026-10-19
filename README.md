# egomlx

A toolkit for building, visualizing, and controlling [gomlx](https://github.com/gomlx/gomlx) neural network models using the [emergent](https://github.com/emer/emergent) GUI, including the [netview](https://github.com/emer/emergent/wiki/NetView) for displaying and interacting with the network.

## Structure

* [`egomlx`](egomlx) -- the core toolkit package: implements the [emer](https://github.com/emer/emergent/wiki/Emer) `Network`, `Layer`, and `Path` interfaces on top of gomlx, providing netview visualization, the standard [params](https://github.com/emer/emergent/wiki/Params) setting system, [looper](https://github.com/emer/emergent/wiki/Looper) integration helpers, and weights I/O.

* [`sims/ra25`](sims/ra25) -- a simple random-associator feedforward network with 2 hidden layers that maps 25 random 5x5 input to output patterns. It serves as the basic template for creating new models, mirroring the [axon/v2/sims/ra25](https://github.com/emer/axon/tree/main/v2/sims/ra25) structure.

## Status

The **forward pass** is fully implemented and live: activations, the output error statistic, and netview visualization all work with real gomlx computation (using the default XLA CPU backend; use `SetBackend` to select another). **Backpropagation and weight updates are currently stubbed**, so the example models do not (yet) learn -- this repository is currently focused on the visualization and control infrastructure, with training support to come.

## Requirements

* Go with the [Cogent Core](https://cogentcore.org/core) GUI framework. See [install](https://www.cogentcore.org/core/setup/install) instructions there.

## Running the example

From `sims/ra25`:

```
$ core run [platform]
```

See the [sims/ra25 README](sims/ra25/README.md) for full instructions.
