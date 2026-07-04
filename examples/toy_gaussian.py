"""Toy demo: two-layer unsupervised PC on a 2-D Gaussian.

The smallest complete PCN: a 1-D latent layer predicts a 2-D input layer
(probabilistic-PCA style). The input is clamped to data drawn from a diagonal
Gaussian; inference relaxes the latent, learning updates the weight, bias, and
per-dimension log-precision of the single Predict connection.

The model is  x = W z + b + eps,  eps ~ N(0, diag(1/precision)). After
training, reconstructions W z* + b recover the data mean, and the learned
per-dimension precision tracks the *residual* error left after the latent
has explained what it can (so the learned sigmas sit below the raw data
sigmas — the 1-D latent soaks up the dominant direction plus some noise).

No dataset download required; runs in a few seconds.
"""
import jax.numpy as jnp
import numpy as np
import optax

import pcn

SEED = 42
N_SAMPLES = 200
BATCH_SIZE = 20
N_EPOCHS = 50
N_ITERS = 10  # inference/learning iterations per batch

TRUE_MEAN = np.array([2.0, 1.0], dtype=np.float32)
TRUE_STD = np.array([1.5, 0.5], dtype=np.float32)


def main():
    rng = np.random.default_rng(SEED)
    X = (rng.standard_normal((N_SAMPLES, 2)) * TRUE_STD + TRUE_MEAN).astype(np.float32)
    loader = [{'x': X[i:i + BATCH_SIZE]} for i in range(0, N_SAMPLES, BATCH_SIZE)]

    net = pcn.PCNetwork(seed=SEED)
    net.config(use_bias=True, learn_precision_weights=False, learn_precision_bias=True)
    with net:
        # Direct (identity) activations: this is a purely linear Gaussian model.
        l_input = pcn.Layer(dim=2, activation=pcn.Direct(), label='input')
        l_hidden = pcn.Layer(dim=1, activation=pcn.Direct(), label='hidden')
        conn = pcn.Predict(l_hidden, l_input)  # hidden predicts input
    net.build()

    sim = pcn.Simulation(net)
    energy_record = {'energy': ((conn.error, conn.precision),
                                lambda e, p: jnp.mean(0.5 * jnp.sum(p * e ** 2 - jnp.log(p))))}

    val_optimizer = optax.sgd(0.5)
    param_optimizer = net.multi_transform(
        {'precision': optax.sgd(1e-2)}, default_optim=optax.adam(1e-2))

    print(f"Data: N(mean={TRUE_MEAN}, std={TRUE_STD}), {N_SAMPLES} samples")
    for epoch in range(N_EPOCHS):
        sim.train(loader, data_map={l_input: 'x'}, epochs=1,
                  iterations_per_sample=0, learning_iterations_per_sample=N_ITERS,
                  record_map=energy_record, verbose=False,
                  params_optimizer=param_optimizer, values_optimizer=val_optimizer)
        if (epoch + 1) % 10 == 0:
            sigma = 1.0 / np.sqrt(np.exp(np.array(sim.params.precision_biases[0])))
            energy = float(np.mean(sim.train_records['energy']))
            print(f"  epoch {epoch + 1:3d} | energy {energy:8.4f} | "
                  f"residual sigma ({sigma[0]:.3f}, {sigma[1]:.3f})")

    # Relax the latent on the full dataset and reconstruct: W z* + b.
    latent_record = {'z': (l_hidden.value, lambda v: np.array(v))}
    results = sim.test([{'x': X}], data_map={l_input: 'x'}, record_map=latent_record,
                       iterations_per_sample=50, verbose=False,
                       values_optimizer=val_optimizer)
    z = np.array(results['z'][0])  # (N, 1)
    W = np.array(sim.params.predict_weights[0])  # (2, 1)
    b = np.array(sim.params.predict_biases[0])   # (2,)
    recon_mean = (z @ W.T + b).mean(axis=0)
    sigma = 1.0 / np.sqrt(np.exp(np.array(sim.params.precision_biases[0])))

    print("\nLearned vs true:")
    print(f"  reconstruction mean: ({recon_mean[0]:+.3f}, {recon_mean[1]:+.3f})   "
          f"true mean ({TRUE_MEAN[0]:+.3f}, {TRUE_MEAN[1]:+.3f})")
    print(f"  residual sigma:      ({sigma[0]:.3f}, {sigma[1]:.3f})   "
          f"data sigma ({TRUE_STD[0]:.3f}, {TRUE_STD[1]:.3f})")
    print("  (residual sigma < data sigma: the latent explains part of each dimension,")
    print("   most of all the wide one; the precision models what is left over)")


if __name__ == "__main__":
    main()
