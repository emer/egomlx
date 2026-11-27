# egomlx

<p align="center">
    <a href="https://pkg.go.dev/github.com/emer/egomlx"><img src="https://img.shields.io/badge/dev-reference-007d9c?logo=go&logoColor=white&style=flat" alt="pkg.go.dev docs"></a>
    <a href="https://github.com/emer/egomlx/actions"><img alt="GitHub Actions Workflow Status" src="https://img.shields.io/github/actions/workflow/status/emer/egomlx/go.yml"></a>
    <a href="https://raw.githack.com/wiki/emer/egomlx/coverage.html"><img alt="Test Coverage" src="https://github.com/emer/egomlx/wiki/coverage.svg"></a>
    <a href="https://github.com/emer/egomlx/tags"><img alt="Version" src="https://img.shields.io/github/v/tag/emer/egomlx?label=version&color=blue"></a>
</p>

A toolkit for building, visualizing, and controlling [gomlx](https://github.com/gomlx/gomlx) neural network models using the [emergent](https://github.com/emer/emergent) GUI, including the [netview](https://github.com/emer/emergent/wiki/NetView) for displaying and interacting with the network.

## Structure

* [`egomlx`](egomlx) -- the core toolkit package: implements the [emer](https://github.com/emer/emergent/wiki/Emer) `Network`, `Layer`, and `Path` interfaces on top of gomlx, providing netview visualization, the standard [params](https://github.com/emer/emergent/wiki/Params) setting system, [looper](https://github.com/emer/emergent/wiki/Looper) integration helpers, and weights I/O.

* [`sims/ra25`](sims/ra25) -- a simple random-associator feedforward network with 2 hidden layers that maps 25 random 5x5 input to output patterns. It serves as the basic template for creating new models, mirroring the [axon/v2/sims/ra25](https://github.com/emer/axon/tree/main/v2/sims/ra25) structure.

## Status

Supports the following features, wrapped in the emer framework:
* **Feedforward** and **recurrent** networks (`NetworkParams.NCycles` > 1) with selectable activation functions, output error statistics, and netview visualization

* **Backward learning pass**: `Network.Backward` builds one compiled graph with forward + loss + autodiff gradients + optimizer weight updates (via [gomlx optimizers](https://pkg.go.dev/github.com/gomlx/gomlx/ml/train/optimizer)). The loss function and optimizer settings are global in `Network.NetworkParams`; pathway-specific learning is controlled by `PathParams.Learn`. DWt (gradient) values are available as a netview synapse variable, read back only when actually being viewed.

* **Recurrent settling**: with `NetworkParams.NCycles` > 1, the network is unrolled for that many cycles, with the weights shared across cycles. `Forward` paths use the sender's current-cycle activity, and the recurrent `Back` and `Lateral` paths use the sender's previous-cycle activity (zero on the first cycle). The loss is computed on the final cycle, so learning is backpropagation through time. Set `Network.RecordCycles` (before `Build`) to also read back every cycle's activity into `Layer.CycleActs` (see `Layer.CycleAct`), for analyzing the settling dynamics.

* `ActParams.Noise` adds Gaussian noise to a layer's net input on every cycle. `NetworkParams.Decay` < 1 carries over the final activity of the previous trial: the first cycle of the next trial starts from (1 − Decay) times that activity, within each data-parallel stream, with no gradient flowing back into the previous trial. `ActParams.Dropout` sets a layer's activity to zero with that probability on each cycle of the training passes only (scaling the rest to keep the mean), with no effect on Forward passes. `ActParams.TestDropout` does the same on the Forward (testing) passes only, with a new random pattern on every pass, so a network can be tested with a different amount of dropout than it was trained with. `ActParams.TestNoise` likewise adds Gaussian noise on the Forward passes only, independent of `Noise` (total SD sqrt(Noise² + TestNoise²) in testing). These test-only perturbations do not affect training: `TrainTrial` runs its forward pass without them (a separate compiled graph, `TrainFwdExec`, created only when they are used), so they also do not reach the activity carried over to the next training trial.

* **Connectivity**: `ConnectLayers` accepts any `paths.Pattern`. Each pathway is a dense gomlx layer, and patterns other than full connectivity (e.g., `paths.PoolOneToOne`) are implemented by multiplying its weights by a fixed 0/1 connectivity mask (`Path.Mask`), so the unconnected weights stay zero and get no gradient. The weights are initialized using the actual number of connected senders (`Path.FanIn`). Note that `paths.Full` leaves out each unit's connection to itself for a lateral (self) pathway, unless its `SelfCon` is set.

## Compute backends

gomlx runs the compiled graphs on a pluggable backend. `compute.MustNew()` picks the backend named by the environment variable `GOMLX_BACKEND`, and uses XLA if it is not set. `Network.SetBackend` sets one explicitly in code.

| `GOMLX_BACKEND` | Hardware (macOS) | Notes |
|---|---|---|
| `xla` (default) | CPU | XLA has no GPU support on macOS. Slowest to compile, fastest per call for small networks. Uses several CPU threads. |
| `mps` | Apple GPU | Metal Performance Shaders, from go-darwinml. No extra install needed. |
| `mlx` | Apple GPU | Apple MLX, from go-darwinml. Needs the MLX library: `brew install mlx`, and then a full rebuild (see below). |

For example:

```sh
GOMLX_BACKEND=mlx ./conflict_rnn -nogui
```

`Network.TimerReport` names the backend in use, so check it to confirm that the one you asked for was picked up.

### Rebuilding after installing MLX

go-darwinml decides whether to include MLX support **at compile time**, by checking whether the MLX headers exist (`__has_include(<mlx/mlx.h>)`). Go's build cache does not notice when those headers are installed later. So a binary built before `brew install mlx` keeps reporting "MLX is not installed", even after rebuilding normally. Force a full rebuild once, which refreshes the cached cgo package:

```sh
brew install mlx
go build -a .
```

After that, ordinary builds of any program pick up the MLX-enabled package from the cache. Do the same after upgrading or removing MLX.

### Timing

`Network.CompileTimer`, `ForwardTimer` and `BackwardTimer` time graph compilation separately from the computation (each `Exec` call, including reading back the results), and `Network.TimerReport` reports all three. Changing the network parameters (`UpdateParams`) or the batch shape rebuilds the graphs, so applying params in bulk (one `UpdateParams` per sheet) keeps the number of compiles down.

For reference, a 40-epoch run of the small recurrent `conflict_rnn` network (10 cycles, batch of 16, on an Apple Silicon Mac) gave identical results on all three backends, with these times:

| Backend | Wall | CPU (user) | Compile (9 graphs) | Forward / call | Backward / call |
|---|---|---|---|---|---|
| XLA | 8.2 s | 14.9 s | 2.17 s | 0.58 ms | 2.17 ms |
| MPS | 11.6 s | 7.8 s | 0.44 s | 1.36 ms | 4.48 ms |
| MLX | 10.9 s | 4.9 s | 0.02 s | 1.62 ms | 4.09 ms |

For small networks and batches like this, the per-call overhead of the GPU backends outweighs their compute advantage, so XLA on the CPU is fastest for a single run. MLX compiles almost instantly and uses the least CPU, so it may be the better choice for running many jobs in parallel. Larger networks and batches should favor the GPU backends.

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

