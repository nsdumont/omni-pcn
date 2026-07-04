"""MNIST discriminative PCN.

The canonical supervised example: a 784-256-256-10 MLP where each layer
predicts the next one up (input -> hidden1 -> hidden2 -> output). During
training both the image and the one-hot label are clamped; at test time only
the image is clamped and inference relaxes the output layer to a prediction.

Reaches ~96% test accuracy in 5 epochs. Downloads MNIST to ./data on first run.
"""
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import pcn

BATCH_SIZE = 256
N_EPOCHS = 5
N_ITERS = 5  # inference/learning iterations per batch


def get_mnist_loaders(batch_size, data_dir="./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    def collate_fn(batch):
        images, labels = zip(*batch)
        images = np.stack([img.numpy().flatten() for img in images])
        one_hot = np.zeros((len(labels), 10), dtype=np.float32)
        one_hot[np.arange(len(labels)), labels] = 1.0
        return {'image': images, 'label': one_hot, 'label_idx': np.array(labels)}

    make = lambda train: DataLoader(
        datasets.MNIST(root=data_dir, train=train, download=True, transform=transform),
        batch_size=batch_size, shuffle=train, collate_fn=collate_fn, drop_last=True)
    return make(True), make(False)


def batch_accuracy(output_values, labels):
    pred = jnp.argmax(output_values, axis=-1)
    true = jnp.argmax(labels, axis=-1)
    return float(jnp.mean(pred == true))


def main():
    print(f"JAX devices: {jax.devices()}")
    train_loader, test_loader = get_mnist_loaders(BATCH_SIZE)

    net = pcn.PCNetwork(seed=10)
    net.config(use_bias=True, learn_precision_weights=False, learn_precision_bias=False)
    with net:
        l_input = pcn.Layer(dim=784, activation=pcn.LeakyRelu(), label="input")
        l_hidden1 = pcn.Layer(dim=256, activation=pcn.LeakyRelu(), label="hidden1")
        l_hidden2 = pcn.Layer(dim=256, activation=pcn.LeakyRelu(), label="hidden2")
        l_output = pcn.Layer(dim=10, activation=pcn.Softmax(), label="output")
        pcn.Predict(l_input, l_hidden1)
        pcn.Predict(l_hidden1, l_hidden2)
        pcn.Predict(l_hidden2, l_output)
    net.build()

    val_optimizer = optax.adam(0.5)
    param_optimizer = optax.chain(optax.add_decayed_weights(1e-4), optax.adam(1e-3))

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
