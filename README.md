# OmniPCN - Omni-Directional Predictive Coding Networks

[![Python](https://img.shields.io/badge/python-3.13-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-EUPL--1.2-blue)](LICENSE)

A JAX-based implementation of Predictive Coding Networks and Graphs for efficient neural network simulation and learning. Install as `pcn` (`import pcn`).

Note that this is a platform for active research. It is subject to change/refactors in the near future. There are several features that add complexity & flexibility in the current backend that are not demonstrated in the examples. These are features for active, unreleased research projects.

## Summary

PCN provides a flexible framework for building and training predictive coding networks. Key features include:

- **Intuitive API**: Define networks using a context manager with `Layer`, `Predict`, `Project`, and `Modulate` connections
- **Flexible Architectures**: Support for discriminative, generative, and bidirectional PC networks and graphs. Linear connections or convolutional.
- **Temporal dynamics**: Networks run over time (per-iteration timesteps via temporal clamping); `Project`/`Modulate` persist as integrating-drive state operators, enabling hand-drafted recurrences
- **Stateful & stochastic neurons**: `Leaky` activations (leaky-integrator errors/precisions) and `Stochastic` activations (noise injection for sampling/generative inference)
- **Multiple Learning Rules**: Can have projections trained with Hebbian, Oja, three-factor Hebbian, or Gradient Descent learning with customizable reward/loss functions (non-PC trained connections alongside the PC ones)
- **Learnable Precision**: Precision can be learned (softplus/exp/linear parameterizations), either as a single bias or as a function of other layer states
- **Composite building blocks**: Reusable groups of layers + connections — `Skip` (delayed identity shortcuts) and `Memory` (Legendre/Laguerre Memory Unit for online history)
- **Optimizers**: Helper functions for multi-transform with optax to set layer and param-type specific optimizers
- **JAX Backend**: GPU and Apple Silicon support



## Installation

### Using uv (recommended)

```bash
# Clone the repository
git clone https://github.com/nsdumont/omni-pcn.git
cd omni-pcn

# Install with uv
uv sync

# For Mac with Metal GPU support
uv sync --group mps

# For NVIDIA GPU support
uv sync --group cuda

# dev + plot groups are installed by default (pytest, ruff, matplotlib, ...)
```

### Using pip

```bash
# Clone and install
git clone https://github.com/nsdumont/omni-pcn.git
cd omni-pcn
pip install -e .

# GPU support and dev tools are managed as dependency groups (PEP 735);
# with pip >= 25.1:
pip install -e . --group cuda   # NVIDIA GPU
pip install -e . --group mps    # Mac (Apple Silicon) GPU
pip install -e . --group dev    # Development (pytest, ruff, ...)
```

## Usage

### Basic Network Definition

```python
import pcn
import jax.numpy as jnp

# Create a discriminative PC network
net = pcn.PCNetwork(seed=42)

with net:
    # Define layers
    l_input = pcn.Layer(dim=784, label="input")
    l_hidden = pcn.Layer(dim=256, activation=pcn.Relu(), label="hidden")
    l_output = pcn.Layer(dim=10, activation=pcn.Softmax(), label="output")

    # Define connections (lower predicts higher for discriminative)
    pcn.Predict(l_input, l_hidden, learning_rate=1e-3)
    pcn.Predict(l_hidden, l_output, learning_rate=1e-3)

# Build the network (compiles structure)
net.build()
```

**Arbitrary graph structure (the "omni-directional" part).** `Predict` connects
any layer(s) to any layer, so you define whatever graph you want. When several
sources feed one layer, *how* you wire them is a choice of generative
factorization — not just an implementation detail:

- **One joint prediction** — pass a list. `Predict([l_image, l_audio], l_shared)`
  concatenates the pre values and predicts the target with a single weight,
  error, and precision: one factor over the joint input. (Layers can also be
  sliced, e.g. `l[0:64]`.)
- **Separate predictions (product-of-experts)** — two edges,
  `Predict(l_image, l_shared)` and `Predict(l_audio, l_shared)`, give the target
  two *independent* predictions, each with its own error and precision. The
  target relaxes to their **precision-weighted combination** — cue integration,
  where each source (or a context signal, via `precision_input`) sets how much
  it is trusted.

```python
# one joint factor: 1 prediction / 1 error / 1 precision
pcn.Predict([l_image, l_audio], l_shared)

# — or a different generative model — product-of-experts: 2 factors
pcn.Predict(l_image, l_shared)     # each source predicts l_shared on its own;
pcn.Predict(l_audio, l_shared)     # l_shared settles at the precision-weighted mix
```

### Optax

```python
val_optimizer = optax.sgd(0.1)
param_optimizer = net.multi_transform(
            {"predict": optax.adam(1e-3), "precision": optax.sgd(1e-5)}, # different for prediction and precision parameters
            default_optim=optax.adam(1e-4),
        )
```

The `"precision"` key routes a separate optimizer to the precision parameters
(often you want a smaller LR than the predict weights).

### Learnable precision

Every `Predict` owns a precision (inverse-variance) on its error. It can be a
learned per-dimension bias, or made **input-dependent** — a function of the pre
activation by default, or of *any other* node(s) via `precision_input`
(a `Layer`/`NodeRef`, possibly sliced, another Predict's `.error`/`.precision`,
or a list of these, concatenated).

```python
with net:
    # How much to trust this prediction is conditioned on a context state,
    # not the input itself — e.g. downweight a sensory error when top-down
    # context signals the input is unreliable.
    pcn.Predict(l_input, l_hidden,
                learn_precision_weights=True,
                precision_input=l_context.value)
```

Value sources are read live (current iteration); error/precision sources use
the previous iteration (Jacobi), so a connection may even read its own error.

### Training with the Simulation Class

```python
# Create and run simulation
sim = pcn.Simulation(net)
sim.train(
    dataloader, # a torch dataloader with dict samples
    data_map={l_input: 'image', l_output: 'label'}, # clamp layers to data keys (see Clamping options)
    epochs=10,
    iterations_per_sample=100, 
    params_optimizer=param_optimizer,
    values_optimizer=val_optimizer,
    verbose=True
)

print(f"Final energy: {sim.final_energy:.4f}")
```

### Clamping options

`data_map` ties layers to sample keys. Beyond the hard clamp
(`{l_input: 'image'}`), three variants:

- **Masked (partial) clamp** — `{l: ('data', 'mask')}` with a 0/1 mask: clamp
  only the observed dimensions and infer the rest. Useful for missing-data /
  inpainting — clamp the known pixels, let inference fill the holes.
- **Soft clamp / nudge** — the same tuple form with a mask in `(0, 1)`: the
  layer is held a fraction `β` toward the data rather than pinned. This is
  somewhat like positive nudging (cf. the PCX paper); in cog-sci terms it lets a prior shape
  the "input" the network actually perceives (as perception partly does).
- **Temporal clamp** — pass a `(batch, T, dim)` array and set
  `iterations_per_sample` to a multiple of `T`; each data timestep then gets
  `iterations_per_sample // T` energy-relaxation iterations while the
  non-clamped latents persist across the sequence. Good for temporal data with
  strong correlations (or clock several relaxation steps per timestep).

```python
sim.train(loader,     data_map={l_input: ('image', 'mask')})       # partial / soft
sim.train(seq_loader, data_map={l_in: 'seq'},                      # (batch, T, dim)
          iterations_per_sample=4 * T)                             # 4 relax steps / timestep
```


### Composite building blocks: `Skip` and `Memory`

Composites are reusable groups of layers and connections created inside the
`with net:` block, just like a single connection.

**`Skip`** — a delayed identity shortcut. Inserts `delay` auxiliary `Direct`
layers chained by fixed-weight (`NoLearning`) identity `Project`s, so a copy of
`pre` reaches `post` (same dim) `delay` timesteps later, scaled by `skip_scale`:

```python
with net:
    l_a = pcn.Layer(dim=64, label="a")
    l_b = pcn.Layer(dim=64, label="b")
    pcn.Skip(l_a, l_b, delay=2, skip_scale=1.0)   # a -> aux1 -> aux2 -> b
```

**`Memory`** — a Legendre/Laguerre Memory Unit (LMU / HiPPO). It maintains an
online polynomial projection of an input signal's recent history as a fixed
linear recurrence `m[t+1] = Ā m[t] + B̄ u[t]`, built entirely from `NoLearning`
`Project`s (no backend changes). Each input channel gets its own
`dims_per_input`-dimensional memory.

```python
with net:
    l_in = pcn.Layer(dim=1, activation=pcn.Direct(), label="in")   # clamp a (B, T, 1) sequence
    mem  = pcn.Memory(l_in, dims_per_input=8,
                      memory_type="legt",   # 'legt'/'lmu' (Legendre) or 'lagt' (Laguerre)
                      theta=1.0,            # window length
                      dt=0.1,               # PER-ITERATION timestep (see notes)
                      mode="deterministic") # vs "energy_coupled"
    l_h = pcn.Layer(dim=32, activation=pcn.Relu(), label="h")
    pcn.Predict(mem.value, l_h)             # learn a readout from the memory state

# reconstruct the input delayed by t in [0, theta] from a memory snapshot:
u_hat = mem.decode(m_state, t=0.5)          # or mem.C(0.5) for the readout matrix
```

- **Modes:** `deterministic` adds an output-mirror layer so energy minimization
  only shapes the readout, leaving the recurrence a pure LMU; `energy_coupled`
  lets readout errors correct the memory.
- **Timing:** the recurrence advances **once per inference iteration**, not per
  input timestep. Feed sequences via temporal clamping (`data_map={l_in: 'seq'}`
  with a `(batch, T, dim)` array) and set `iterations_per_sample = T`. `dt` is
  per-iteration — if `total_iterations ≠ T`, scale `dt` by `iters_per_timestep`.
  Use `feedforward_init=False` to avoid an off-by-one in the memory state.
- `decode`/`C` are exact for the Legendre (`legt`) basis.

### Stateful & stochastic activations

Beyond the stateless activations (`Direct`, `Relu`, `Tanh`, `Sigmoid`,
`Softmax`, `Gelu`, `Elu`, `LeakyRelu`), two wrappers add temporal/stochastic
behaviour and can be passed wherever an activation is accepted (layer, or a
`Predict`'s `error_activation` / `precision_activation`):

**`Leaky(base, leak)`** — a leaky integrator: `y_t = (1-leak)·base(x_t) + leak·y_{t-1}`.
Mainly for error/precision nodes (values already integrate via their carried
state). `leak=0` recovers `base`.

```python
with net:
    l_x = pcn.Layer(dim=20, label="x")
    l_y = pcn.Layer(dim=10, label="y")
    pcn.Predict(l_x, l_y,
                error_activation=pcn.Leaky(pcn.Direct(), leak=0.3),     # leaky error
                precision_activation="softplus")
```

**`Stochastic(base, noise_fn=gaussian_noise, **noise_params)`** — additive
sampled noise: `y = base(x) + noise`. Use as a layer activation to inject noise
into predictions (sampling / generative inference). The backend folds a fresh
PRNG key per iteration/node; with no key (e.g. logging) it falls back to the
deterministic `base` output.

```python
with net:
    l_latent = pcn.Layer(dim=16, activation=pcn.Stochastic(pcn.Tanh(), sigma=0.1), label="z")
# pass is_stochastic=True to sim.train / sim.test to sample predictions during inference
```

### Examples

Self-contained example scripts live in [examples/](examples/) — a toy
Gaussian demo (no dataset needed), discriminative / generative / bidirectional
MNIST, and a convolutional CIFAR-10 classifier:

```bash
uv run python examples/mnist_discriminative.py
```

## Performance

Comparison to four public predictive-coding libraries on a discriminative MLP network with layer dims `784 → 500×3 → 10`, batch 256, `T` value-inference steps then one weight update (not iPC). A single NVIDIA RTX 5090 (32 GB, CUDA) on Linux was used. Each library was run in a different conda env for the different requirements. OmniPCN on
`jax 0.9.2`; PCX (liukidar/pcx) on `jax 0.4.38`; JPC (thebuckleylab/jpc) on
`jax 0.5.2`; PRECO (bjornvz/PRECO, the ACM survey tutorial code) on `torch 2.11`;
pcn-torch (anonx3247/pcn-torch) on `torch 2.11`. Numbers are the
median of 3×50-batch runs, each in a fresh process; peak memory is the device
allocator's `peak_bytes_in_use`.

| Library | ms / inference iter (no overhead) | train step, T=20 (ms/batch; includes overhead) | peak mem, width 500 (MiB) | peak mem, width 2048 (MiB) |
|---|--:|--:|--:|--:|
| **OmniPCN** | 0.09 | 2.47 | 163 | 261 |
| PCX | 0.10 | 2.14 | 156 | 538 |
| PRECO | 0.12 | 3.00 | 71 | 292 |
| JPC | 0.06  | 11.94 | 97 | 603 |
| pcn-torch | —  | 945  | 16 | 97 |

Note that pcn-torch is unbatched so its "train step" is
`256 × 3.7 ms/sample`  hence tiny memory but ~300×
the wall-clock. JPC has a higher overhead so the per-iteration cost is low but the full train step time is higher. 

Overall the three cluster closely on the full train step, with PCX slightly ahead on its very low fixed overhead; OmniPCN has the lowest *per-inference-iteration* cost of the batched libraries (0.09 ms, below PCX's 0.10) so it pulls even or ahead at higher iteration counts and greater depth. OmniPCN adds learnable precision and easy-to-set-up arbitrary-graph wiring the others do not have; PRECO (the ACM survey's tutorial code) is a lean fixed-MLP/graph implementation; JPC offers different ODE solver backends which sets it apart.

On memory, PRECO is the leanest at typical widths (plain fp32 PyTorch with no autograd graph or optimizer state), while OmniPCN scales the best with width — its donate-based buffer reuse keeps growth low, so it overtakes the others by width 2048, at the cost of a larger fixed footprint at small width from its learnable-precision arrays.

## High level architecture

```
pcn/
├── core/           # Network definition (OOP API)
│   ├── network.py      # PCNetwork context manager
│   ├── layer.py        # Layer and NodeRef
│   ├── connections.py  # Predict, Project, Modulate
│   └── ...
├── backend/        # JAX-compiled functions
│   └── simulation.py   # run_batch + helpers
└── simulation.py   # High-level Simulation class
```

## Full repository structure

```
omnipcn/
├── pcn/                            # Core library package
│   ├── __init__.py                 # Public API exports
│   ├── simulation.py               # High-level Simulation class
│   ├── backprop_simulation.py      # Backprop baseline for comparison
│   ├── config.py                   # load_config(), _DEFAULT sentinel, DEFAULTS
│   ├── default_config.json         # Single source of truth for all defaults
│   ├── core/                       # Network definition (OOP API)
│   │   ├── network.py              # PCNetwork context manager
│   │   ├── layer.py                # Layer, NodeRef
│   │   ├── connections.py          # Predict, PredictRes, PredictConv, PredictTransConv, Project, Modulate
│   │   ├── skip.py                 # Skip composite (delayed identity shortcuts)
│   │   ├── memory.py               # Memory (LMU / HiPPO) composite
│   │   ├── activations.py          # Activation classes + ACTIVATION_REGISTRY
│   │   ├── learning_rules.py       # Hebbian, Oja, ThreeFactorHebbian, GradientDescent, NoLearning
│   │   ├── optimizers.py           # Optimizer helpers (natural gradient, ...)
│   │   ├── regularization.py       # L2Norm, SIGReg regularizers
│   │   ├── state.py                # NetworkState, NetworkParams
│   │   └── structure.py            # NetworkStructure (compiled graph)
│   ├── backend/                    # JAX-compiled functions
│   │   ├── simulation.py           # run_batch, _inference_step, _combined_step
│   │   ├── backprop_simulation.py  # Backprop backend
│   │   └── rewards.py              # Reward functions for three-factor Hebbian
│   └── tests/                      # Test suite (pytest)
│
├── examples/                       # Self-contained example scripts
│   ├── toy_gaussian.py             # Smallest complete PCN (no dataset needed)
│   ├── mnist_discriminative.py     # Supervised classification
│   ├── mnist_generative.py         # Top-down generation + classification
│   ├── mnist_bidirectional.py      # Joint bidirectional wiring
│   └── cifar10_conv.py             # Convolutional PCN
│
├── pyproject.toml                  # Project config and dependencies
├── uv.lock                         # Locked dependency versions
└── LICENSE
```

## Testing

```bash
uv run pytest pcn/tests/ -v
```

## License

This project is licensed under the European Union Public Licence (EUPL) v1.2 - see the [LICENSE](LICENSE) file for details.


