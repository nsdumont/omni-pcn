"""CIFAR-10 convolutional discriminative PCN.

Four strided PredictConv stages (VGG-5 channel widths 128/256/512/512, all
3x3 kernels, stride 2) downsample 3x32x32 images to a 512-channel 2x2 map,
followed by a dense Predict to the 10-way output. Layers hold flattened
feature maps; PredictConv handles the (channels, H, W) reshape internally, so
each conv layer's dim must be out_channels * H_out * W_out — computed here
with a small helper that mirrors PredictConv's shape logic.

Two tuning notes that matter on conv PCNs: weight decay is destructive here
(it suppresses train and test accuracy together), and data augmentation
underfits at short training budgets — both are off. Expect roughly 58% test
accuracy after 10 epochs; with horizontal-flip augmentation, a cosine LR
schedule, and ~80 epochs this architecture reaches ~65%.

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

BATCH_SIZE = 256
N_EPOCHS = 10
N_ITERS = 10  # inference/learning iterations per batch

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_cifar10_loaders(batch_size, data_dir="./data"):
    transform_train = transforms.Compose([
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


def conv_output_dim(input_shape, out_channels, kernel_size, stride, padding):
    """Output spatial shape and flat dim of a PredictConv post layer."""
    h, w = input_shape
    h_out = (h + 2 * padding - kernel_size) // stride + 1
    w_out = (w + 2 * padding - kernel_size) // stride + 1
    return (h_out, w_out), out_channels * h_out * w_out


def batch_accuracy(output_values, labels):
    pred = jnp.argmax(output_values, axis=-1)
    true = jnp.argmax(labels, axis=-1)
    return float(jnp.mean(pred == true))


def main():
    print(f"JAX devices: {jax.devices()}")
    train_loader, test_loader = get_cifar10_loaders(BATCH_SIZE)

    s1_shape, s1_dim = conv_output_dim((32, 32), 128, kernel_size=3, stride=2, padding=1)
    s2_shape, s2_dim = conv_output_dim(s1_shape, 256, kernel_size=3, stride=2, padding=1)
    s3_shape, s3_dim = conv_output_dim(s2_shape, 512, kernel_size=3, stride=2, padding=1)
    s4_shape, s4_dim = conv_output_dim(s3_shape, 512, kernel_size=3, stride=2, padding=1)

    net = pcn.PCNetwork(seed=10)
    net.config(use_bias=True, learn_precision_weights=False, learn_precision_bias=False)
    with net:
        l_input = pcn.Layer(dim=3 * 32 * 32, activation=pcn.Direct(), label="input")
        l_conv1 = pcn.Layer(dim=s1_dim, activation=pcn.LeakyRelu(), label="conv1")
        l_conv2 = pcn.Layer(dim=s2_dim, activation=pcn.LeakyRelu(), label="conv2")
        l_conv3 = pcn.Layer(dim=s3_dim, activation=pcn.LeakyRelu(), label="conv3")
        l_conv4 = pcn.Layer(dim=s4_dim, activation=pcn.LeakyRelu(), label="conv4")
        l_output = pcn.Layer(dim=10, activation=pcn.Softmax(), label="output")

        pcn.PredictConv(l_input, l_conv1, kernel_size=3, input_shape=(32, 32),
                        stride=2, padding=1)
        pcn.PredictConv(l_conv1, l_conv2, kernel_size=3, input_shape=s1_shape,
                        stride=2, padding=1)
        pcn.PredictConv(l_conv2, l_conv3, kernel_size=3, input_shape=s2_shape,
                        stride=2, padding=1)
        pcn.PredictConv(l_conv3, l_conv4, kernel_size=3, input_shape=s3_shape,
                        stride=2, padding=1)
        pcn.Predict(l_conv4, l_output)
    net.build()
    print(f"Layer dims: {list(net.structure.layer_dims)}")

    val_optimizer = optax.adam(0.5)
    param_optimizer = optax.adam(5e-4)  # no weight decay: destructive on conv PCNs

    sim = pcn.Simulation(net)
    record = {'batch_accuracy': ((l_output.value, 'label'), batch_accuracy)}

    for epoch in range(N_EPOCHS):
        t0 = time.perf_counter()
        sim.train(train_loader, data_map={l_input: 'image', l_output: 'label'}, epochs=1,
                  iterations_per_sample=0, learning_iterations_per_sample=N_ITERS,
                  verbose=False,
                  params_optimizer=param_optimizer, values_optimizer=val_optimizer)
        results = sim.test(test_loader, data_map={l_input: 'image'}, record_map=record,
                           iterations_per_sample=N_ITERS, verbose=False,
                           values_optimizer=val_optimizer)
        acc = np.mean(results['batch_accuracy'])
        print(f"Epoch {epoch + 1}/{N_EPOCHS} | test accuracy {acc * 100:.2f}% | "
              f"{time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
