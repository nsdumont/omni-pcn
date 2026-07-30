# Examples

Each example is a single self-contained script — run it from this directory
(or anywhere; datasets download to `./data` relative to your working
directory):

```bash
uv run python examples/mnist_discriminative.py
```

| Script | What it shows |
|---|---|
| [toy_gaussian.py](toy_gaussian.py) | Smallest complete PCN: 1-D latent predicts 2-D Gaussian data; learns mean, weights, and per-dim precision. No dataset needed. |
| [mnist_discriminative.py](mnist_discriminative.py) | Supervised classification: bottom-up prediction chain, ~96% test accuracy in 5 epochs. |
| [mnist_generative.py](mnist_generative.py) | Top-down prediction chain: one network classifies (clamp image, infer label) and generates (clamp label, infer image). |
| [mnist_bidirectional.py](mnist_bidirectional.py) | Bidirectional wiring: separate bottom-up (discriminative) and top-down (generative) connections; classifies ~84% and generates legible digits from the same weights. |
| [cifar10_conv.py](cifar10_conv.py) | Convolutional PCN on CIFAR-10: VGG-5 built using the  `PredictConvPool` (conv+maxpool) connection, trained with the PCX-benchmark iPC hyperparameters. ~76% test accuracy in 10 epochs, ~82% at the default 50 (PCX reference for this arch: 85.5%). |
| [temporal_pendulum.py](temporal_pendulum.py) | Temporal PC on the noisy-pendulum tracking task from Fig. 7 of Millidge et al. 2024, *Predictive coding networks for temporal prediction*: a two-layer tPC learns transition + observation weights online, in one pass, from noisy observations. Similar model to the paper's, with three changes — the canonical `z[t] = W tanh(z[t-1])` transition via a delayed self-edge (`Predict(z, z, delay=1, delay_unit='timestep')`) instead of their Euler-residual form, a soft precision-weighted temporal prior (π_p:π_o = 5:1, relaxed jointly) instead of their hard prior-reset, and K=12 combined inference+learning iterations per frame. One-step prediction error ≈0.029 vs ≈0.082 for the paper's nonlinear model and 0.061 for copy-last-frame persistence; prints 1/5/10-step errors and saves a phase portrait + animated linkage GIF. No dataset needed. |
| [temporal_double_pendulum.py](temporal_double_pendulum.py) | Hierarchical temporal PC learning the flow map of a *chaotic* double pendulum (slightly unequal arms). Two latent levels, each with a delayed self-edge, joined by a top-down Predict; trained online on 64 random-initial-condition trajectories (30 passes) with a strong transition prior (π_p:π_o = 10:1). Evaluated on held-out trajectories with **periodic clamping**: the state is given only every 12 frames via a temporal mask, and the batch dimension phase-shifts the clamp grid, so one batched `sim.test` yields the full look-ahead curve h=1..9. Prediction error stays 1.6–9× below persistence and *saturates* with h while persistence keeps growing; prints the curve and saves a look-ahead plot + animated linkage GIF. No dataset needed. |
