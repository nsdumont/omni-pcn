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
