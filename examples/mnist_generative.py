"""MNIST generative PCN.

The prediction direction is reversed relative to the discriminative example:
the label layer sits at the top and predicts down through the hiddens to the
image (label -> hidden2 -> hidden1 -> image). Training clamps both ends. The
same trained network then does two things:

  1. Classify: clamp an image, let inference relax the label layer.
  2. Generate: clamp a one-hot label, let inference relax the image layer.

Saves a figure of one generated digit per class next to this script.
Downloads MNIST to ./data on first run.
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import pcn

BATCH_SIZE = 256
N_EPOCHS = 5
N_ITERS = 5        # inference/learning iterations per training batch
N_CLS_ITERS = 20   # relaxation to classify (clamp image, infer label)
N_GEN_ITERS = 100  # longer relaxation to generate (clamp label, infer image)

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


def get_mnist_loaders(batch_size, data_dir="./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
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
        l_image = pcn.Layer(dim=784, activation=pcn.LeakyRelu(), label="image")
        l_hidden1 = pcn.Layer(dim=256, activation=pcn.LeakyRelu(), label="hidden1")
        l_hidden2 = pcn.Layer(dim=256, activation=pcn.LeakyRelu(), label="hidden2")
        # Direct (not Softmax) label activation: softmax squashes a clamped
        # one-hot to a nearly uniform vector, starving the top-down pathway of
        # class contrast. The raised precision on the label connection makes
        # the class drive dominate inference during generation.
        l_label = pcn.Layer(dim=10, activation=pcn.Direct(), label="label")
        pcn.Predict(l_label, l_hidden2, init_log_precision=2.1)
        pcn.Predict(l_hidden2, l_hidden1)
        pcn.Predict(l_hidden1, l_image)
    net.build()

    # Two value optimizers: large-step sgd during the short training
    # relaxations (this is what makes the learned weights generative), adam for
    # the long test-time relaxations where sgd(100) would overshoot.
    train_val_optimizer = optax.sgd(100.)
    test_val_optimizer = optax.adam(0.5)
    param_optimizer = optax.chain(optax.add_decayed_weights(1e-4), optax.adam(1e-3))

    sim = pcn.Simulation(net)
    acc_record = {'batch_accuracy': ((l_label.value, 'label'), batch_accuracy)}

    for epoch in range(N_EPOCHS):
        sim.train(train_loader, data_map={l_image: 'image', l_label: 'label'}, epochs=1,
                  iterations_per_sample=0, learning_iterations_per_sample=N_ITERS,
                  verbose=False,
                  params_optimizer=param_optimizer, values_optimizer=train_val_optimizer)
        # Classification with a generative net: clamp the image, infer the label.
        results = sim.test(test_loader, data_map={l_image: 'image'}, record_map=acc_record,
                           iterations_per_sample=N_CLS_ITERS, verbose=False,
                           values_optimizer=test_val_optimizer)
        acc = np.mean(results['batch_accuracy'])
        print(f"Epoch {epoch + 1}/{N_EPOCHS} | test accuracy {acc * 100:.2f}%")

    # Generation: clamp each one-hot label, relax, read the image layer through
    # its activation.
    f_image = pcn.backend.ACTIVATIONS[l_image.f.type_id]
    gen_record = {'image': (l_image.value, lambda v: f_image(np.array(v)))}
    viz = sim.test([{'label': np.eye(10, dtype=np.float32)}], data_map={l_label: 'label'},
                   record_map=gen_record, iterations_per_sample=N_GEN_ITERS,
                   verbose=False, values_optimizer=test_val_optimizer)
    generated = np.array(viz['image'][0])  # (10, 784)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 10, figsize=(15, 2))
    for c in range(10):
        img = generated[c] * MNIST_STD + MNIST_MEAN
        axes[c].imshow(img.reshape(28, 28), cmap='gray')
        axes[c].set_title(str(c))
        axes[c].axis('off')
    out = Path(__file__).parent / "mnist_generative_samples.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved generated digits to {out}")


if __name__ == "__main__":
    main()
