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
| [cifar10_conv.py](cifar10_conv.py) | Convolutional PCN: four `PredictConv` stages + dense head on CIFAR-10 (VGG-5 widths; ~58% test accuracy in 10 epochs, ~65% with flip augmentation + cosine LR + 80 epochs). |

## Reference: PCX VGG-5 iPC on CIFAR-10

[cifar10_conv.py](cifar10_conv.py) uses the same VGG-5 widths as the CIFAR-10
benchmark in [PCX](https://github.com/liukidar/pcx) (Pinchetti et al., ICLR
2025, *Benchmarking Predictive Coding Networks — Made Simple*), so their tuned
**iPC** (incremental PC) recipe is a useful target. Their reported number is
**85.51 ± 0.12 %** test accuracy (the best PC-family result on that benchmark is
a *centered-nudging* variant at 89.47 %, essentially matching BP-SE's 89.43 %).

**Architecture (VGG-5, 5 weight layers).** Input 3×32×32; four conv blocks each
`Conv(3×3, pad 1) → act → MaxPool(2,2)`, then a linear head. An activity node
(`Vode`) sits after every block; the output node uses a squared-error energy and
is label-clamped.

| Block | Conv | Spatial out |
|---|---|---|
| 1 | 3 → 128 | 128 × 16 × 16 |
| 2 | 128 → 256 | 256 × 8 × 8 |
| 3 | 256 → 512 | 512 × 4 × 4 |
| 4 | 512 → 512 | 512 × 2 × 2 |
| head | Linear 2048 → 10 | 10 |

(`cifar10_conv.py` downsamples with stride-2 convs. To reproduce the VGG-5
`conv → pool` blocks faithfully, use the fused pooling transforms —
`pcn.PredictConvPool(pre, post, kernel_size, input_shape, pool='max'|'avg',
pool_size=2)`, or `transformation='conv-maxpool'` / `'conv-avgpool'` on a plain
`Predict`. Each is one learnable edge computing `pool(conv(f(pre)))`, with the
error's feedback to the pre value routed through the pool adjoint — argmax
unpooling for max, uniform upsampling for avg — automatically. Fusing the pool
into the conv edge (rather than giving the pooled map its own activity node)
keeps the PC inference depth the same as the strided variant. Avgpool is the
smooth, safe default; maxpool's argmax routing is non-smooth and can make value
inference noisier, so benchmark it against stride-2 rather than assuming a win.)

**Training schedule — what makes it *iPC*.** Standard PC runs `T` activity-
inference steps and then does **one** weight update at the relaxed state. iPC
instead updates **the weights on every inference step** — activity and weight
gradients are taken together each of the `T` iterations, giving `T` weight
updates per batch. The AdamW LR schedule is therefore stretched by `T`
(warmup/decay counted in `len(loader)·epochs·T` steps, not `·epochs`).

**Hyperparameters (from `VGG5_iPC.yaml`; Optuna-tuned, so the odd decimals):**

| Field | Value |
|---|---|
| T (inference steps = weight updates/batch) | 8 |
| activation | `hard_tanh` |
| batch size | 128 |
| epochs | 50 |
| weight optimizer | AdamW, lr ≈ 1.75e-4, wd ≈ 2.6e-5 |
| weight LR schedule | warmup-cosine: peak 1.1×lr, warmup 10 % of `steps·T`, decay to 0.1×lr |
| activity optimizer | SGD, lr ≈ 0.744, momentum 0.65 |
| label clamp β | 1.0 (hard clamp; no nudge schedule) |
| augmentation | RandomHorizontalFlip(0.5), RandomCrop(32, pad 4) |
| normalization | mean (0.4914, 0.4822, 0.4465), std (0.2023, 0.1994, 0.2010) |

In OmniPCN, iPC's "learn every inference step" corresponds to running
`Simulation.train` with `iterations_per_sample=0` and
`learning_iterations_per_sample=T` (simultaneous inference + learning), rather
than the default `iterations_per_sample=T, learning_iterations_per_sample=0`
(relax first, then a single update).

### Positive nudging (PN) via soft-clamping in OmniPCN

PCX's *nudging* variants replace the hard output clamp with a partial one:
instead of pinning the output to the label, they hold it a fraction `β` of the
way there — `h = u − β·(u − y)` — so the output is only softly attracted to the
label `y` (β = 1 recovers the hard clamp, β → 0 is a free output; PCX's PN grows
β from ~0.32 by +0.02/epoch).

OmniPCN's analogue is **soft-clamping**: pass a `(data, mask)` pair in
`data_map` and the clamp becomes a per-element blend rather than a hard pin.

```python
# hard clamp (standard PC-SE):       data_map = {l_output: 'label'}
# soft clamp / nudge (strength β):    data_map = {l_output: ('label', 'label_mask')}
#   where the batch dict also carries 'label_mask' = full((B, 10), beta), beta ∈ [0, 1]
```

How the blend works: at the feed-forward init the output is set to
`β·y + (1 − β)·u` — exactly PN's `h = u − β·(u − y)`. Through the following `T`
inference steps the layer keeps a fraction `β` of its value each step
(`v ← v − (1 − β)·lr·∂E/∂v`), i.e. its relaxation toward the network's own
prediction is damped by `(1 − β)`, so for the short `T` used in practice it
stays held near the label. Net effect is the same knob as PCX nudging: `β = 1`
hard clamp ↔ `β = 0` free output, with intermediate `β` a partial hold. Raise
the `label_mask` fill value each epoch to mirror PCX's per-epoch β schedule.

Caveat on the correspondence: PCX's PN adds a *standing* label-attracting
energy term of strength β (fixed point at a `β`-weighted blend of label and
prediction), whereas OmniPCN's soft-clamp holds the label-initialised value
with a per-step damping (its untruncated fixed point is the free prediction).
They coincide at init and behave equivalently in the finite-`T` regime PC
actually runs in, but are not identical dynamical systems — for an exact
standing β-nudge, add a low-precision `Predict` edge from a label node into the
output (precision ∝ β²) instead. Empirically the soft-clamp knob (β ≈ 0.5
output nudge) gives a mild gain for generative/bidirectional nets while a hard
clamp stays best for pure discrimination.
