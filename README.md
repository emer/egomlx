# egomlx

A toolkit for building, visualizing, and controlling [gomlx](https://github.com/gomlx/gomlx) neural network models using the [emergent](https://github.com/emer/emergent) GUI, including the [netview](https://github.com/emer/emergent/wiki/NetView) for displaying and interacting with the network.

## Structure

* [`egomlx`](egomlx) -- the core toolkit package: implements the [emer](https://github.com/emer/emergent/wiki/Emer) `Network`, `Layer`, and `Path` interfaces on top of gomlx, providing netview visualization, the standard [params](https://github.com/emer/emergent/wiki/Params) setting system, [looper](https://github.com/emer/emergent/wiki/Looper) integration helpers, and weights I/O.

* [`sims/ra25`](sims/ra25) -- a simple random-associator feedforward network with 2 hidden layers that maps 25 random 5x5 input to output patterns. It serves as the basic template for creating new models, mirroring the [axon/v2/sims/ra25](https://github.com/emer/axon/tree/main/v2/sims/ra25) structure.

## Status

The **forward pass** is fully implemented and live: activations, the output error statistic, and netview visualization all work with real gomlx computation (using the default XLA CPU backend; use `SetBackend` to select another). **The backward learning pass is also live**: `Network.Backward` builds one compiled graph with forward + loss + autodiff gradients + optimizer weight updates (via [gomlx optimizers](https://pkg.go.dev/github.com/gomlx/gomlx/ml/train/optimizer)), so the example models genuinely learn. The loss function and optimizer settings are global in `Network.NetworkParams`; pathway-specific learning is controlled by `PathParams.Learn`. DWt (gradient) values are available as a netview synapse variable, read back only when actually being viewed.

## Requirements

* Go with the [Cogent Core](https://cogentcore.org/core) GUI framework. See [install](https://www.cogentcore.org/core/setup/install) instructions there.

## Running the example

From `sims/ra25`:

```
$ core run [platform]
```

See the [sims/ra25 README](sims/ra25/README.md) for full instructions.

## Use of AI

This is coded largely using AI tools, under strong human supervision.

