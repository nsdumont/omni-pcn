"""CIFAR-10 convolutional PCN — VGG-5 with fused conv+maxpool blocks.

The architecture and training recipe follow the tuned iPC configuration from
the PCX benchmark (Pinchetti et al., ICLR 2025; see examples/README.md). Four
VGG blocks, each a single learnable `PredictConvPool` edge computing
`maxpool(conv3x3(f(pre)))`, then a dense head:

    3x32x32 -> 128x16x16 -> 256x8x8 -> 512x4x4 -> 512x2x2 -> 10

Three ingredients matter far more than the architecture:

1. iPC: weights update on every one of the T=8 inference iterations
   (`iterations_per_sample=0, learning_iterations_per_sample=T`), with the
   AdamW warmup-cosine schedule stretched accordingly (steps = batches*epochs*T).
2. Value inference by SGD+momentum (not Adam), with each connection's
   precision initialised to its output dimension: the backend energy averages
   over feature dims, so this makes every layer relax at the same rate
   regardless of size (sum-over-dims convention). Without it, large conv
   layers barely move and accuracy drops ~7pp.
3. hard_tanh states and a hard label clamp.

Expect ~76% test accuracy at 10 epochs, ~82% at the default 50 epochs
(~10 min on an RTX 5090; the PCX reference for this arch/recipe is 85.5%).
Downloads CIFAR-10 to ./data on first run.
"""
import os
import sys

if sys.platform == "linux":
    # The default XLA GPU allocator fragments under the many differently-shaped
    # buffers of alternating conv-PCN train/test phases, causing sporadic very
    # slow epochs; the CUDA async allocator avoids this.
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import pcn

BATCH_SIZE = 128
N_EPOCHS = 50
T = 8            # inference iterations per batch = weight updates per batch (iPC)
LR_W = 1.75e-4   # AdamW peak is 1.1x this; warmup 10%, decay to 0.1x
WD = 2.6e-5
LR_H = 0.744     # value-inference SGD step
MOMENTUM_H = 0.65

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def get_cifar10_loaders(batch_size, data_dir="./data"):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    def collate_fn(batch):
        images, labels = zip(*batch)
        images = np.stack([img.numpy().reshape(-1) for img in images])
        one_hot = np.zeros((len(labels), 10), dtype=np.float32)
        one_hot[np.arange(len(labels)), labels] = 1.0
        return {'image': images, 'label': one_hot, 'label_idx': np.array(labels)}

    train_loader = DataLoader(
        datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train),
        batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    test_loader = DataLoader(
        datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test),
        batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last=True)
    return train_loader, test_loader


def batch_accuracy(output_values, labels):
    pred = jnp.argmax(output_values, axis=-1)
    true = jnp.argmax(labels, axis=-1)
    return float(jnp.mean(pred == true))


def main():
    print(f"JAX devices: {jax.devices()}")
    train_loader, test_loader = get_cifar10_loaders(BATCH_SIZE)

    net = pcn.PCNetwork(seed=10)
    net.config(use_bias=True, learn_precision_weights=False, learn_precision_bias=False)
    channels = [128, 256, 512, 512]
    with net:
        l_input = pcn.Layer(dim=3 * 32 * 32, activation=pcn.Direct(), label="input")
        prev, (h, w) = l_input, (32, 32)
        for k, c in enumerate(channels):
            h, w = h // 2, w // 2  # conv k3 s1 p1 keeps size; the 2x2 maxpool halves it
            dim = c * h * w
            l_conv = pcn.Layer(dim=dim, activation=pcn.HardTanh(), label=f"conv{k + 1}")
            # init_precision = output dim -> sum-over-dims value dynamics (see docstring)
            pcn.PredictConvPool(prev, l_conv, kernel_size=3, input_shape=(h * 2, w * 2),
                                pool='max', pool_size=2, stride=1, padding=1,
                                init_precision=float(dim))
            prev = l_conv
        l_output = pcn.Layer(dim=10, activation=pcn.Direct(), label="output")
        pcn.Predict(prev, l_output, init_precision=10.0)
    net.build()
    print(f"Layer dims: {list(net.structure.layer_dims)}")

    # The param optimizer steps once per LEARNING ITERATION (T times per batch),
    # so the schedule horizon is batches * epochs * T.
    total_steps = len(train_loader) * N_EPOCHS * T
    sched = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=1.1 * LR_W,
        warmup_steps=int(0.10 * total_steps), decay_steps=total_steps,
        end_value=0.1 * LR_W)
    param_optimizer = optax.adamw(sched, weight_decay=WD)
    val_optimizer = optax.sgd(LR_H, momentum=MOMENTUM_H)

    sim = pcn.Simulation(net)
    record = {'batch_accuracy': ((l_output.value, 'label'), batch_accuracy)}
    # Evaluation is a pure feedforward pass: one iteration with a zero-LR value
    # optimizer leaves every state at its feedforward init.
    eval_opt = optax.sgd(0.0)

    best = 0.0
    for epoch in range(N_EPOCHS):
        t0 = time.perf_counter()
        sim.train(train_loader, data_map={l_input: 'image', l_output: 'label'}, epochs=1,
                  iterations_per_sample=0, learning_iterations_per_sample=T,
                  verbose=False,
                  params_optimizer=param_optimizer, values_optimizer=val_optimizer)
        results = sim.test(test_loader, data_map={l_input: 'image'}, record_map=record,
                           iterations_per_sample=1, verbose=False,
                           values_optimizer=eval_opt)
        acc = float(np.mean(results['batch_accuracy']))
        best = max(best, acc)
        print(f"Epoch {epoch + 1}/{N_EPOCHS} | test accuracy {acc * 100:.2f}% "
              f"(best {best * 100:.2f}%) | {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
