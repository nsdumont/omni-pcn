# OmniPCN - Omni-Directional Predictive Coding Networks

[![Python](https://img.shields.io/badge/python-3.13-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-EUPL--1.2-blue)](LICENSE)

A JAX-based implementation of Predictive Coding Networks and Graphs for efficient neural network simulation and learning. Install as `pcn` (`import pcn`).

## Summary

PCN provides a flexible framework for building and training predictive coding networks. Key features include:

- **Intuitive API**: Define networks using a context manager with `Layer`, `Predict`, `Project`, and `Modulate` connections
- **JAX Backend**: Automatic GPU/Apple Silicon support
- **Flexible Architectures**: Support for discriminative, generative, and bidirectional PC networks and graphs. Linear connections or convolutional.
- **Composite building blocks**: Reusable groups of layers + connections — `Skip` (delayed identity shortcuts) and `Memory` (Legendre/Laguerre Memory Unit for online history) — drop straight into a `with net:` block
- **Temporal dynamics**: Networks run over time (per-iteration timesteps via temporal clamping); `Project`/`Modulate` persist as integrating-drive state operators, enabling hand-drafted recurrences
- **Stateful & stochastic neurons**: `Leaky` activations (leaky-integrator errors/precisions) and `Stochastic` activations (noise injection for sampling/generative inference)
- **Multiple Learning Rules**: Can have projections trained with Hebbian, Oja, three-factor Hebbian, or Gradient Descent learning with customizable reward/loss functions (non-PC trained)
- **Learnable Precision**: Precision can be learned (softplus/exp/linear parameterizations), either as a single bias or as a function of other layer states
- **Optimizers**: Helper functions for multi-transform with optax to set layer and param-type specific optimizers
- **Hyperparameter optimization**: Helper script for hyperparameter optimization 


## Installation

### Using uv (recommended)

```bash
# Clone the repository
git clone https://github.com/nsdumont/omnipcn.git
cd omnipcn

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
git clone https://github.com/nsdumont/omnipcn.git
cd omnipcn
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

### Optax

```python
val_optimizer = optax.sgd(0.1)
param_optimizer = net.multi_transform(
            {"predict": optax.adam(1e-3), "precision": optax.sgd(1e-5)}, # different for prediction and precision parameters
            default_optim=optax.adam(1e-4),
        )
```

### Training with the Simulation Class

```python
# Create and run simulation
sim = pcn.Simulation(net)
sim.train(
    dataloader, # a torch dataloader with dict samples
    data_map={l_input: 'image', l_output: 'label'}, # clamp these layers with keys from the data
    # data_map={l_input: ('image','mask'), l_output: 'label'}, # support for partial clamping using a mask in the data
    epochs=10,
    iterations_per_sample=100, 
    params_optimizer=param_optimizer,
    values_optimizer=val_optimizer,
    verbose=True
)

print(f"Final energy: {sim.final_energy:.4f}")
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


