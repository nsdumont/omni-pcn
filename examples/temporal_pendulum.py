"""Temporal PC: online tracking and prediction of a noisy pendulum.

The task from Fig. 7 of Millidge et al. 2024, "Predictive coding networks for
temporal prediction": a swinging pendulum (state = angle theta_1 and angular
velocity theta_2) is observed once per frame under Gaussian noise, and a
two-layer temporal PCN must track it and predict the next observation, learning
its transition and observation weights online in a single pass.

The model here differs from the paper's in three ways:
  - canonical tPC transition  z[t] = W tanh(z[t-1])  via a delayed self-edge
    ``Predict(z, z, delay=1, delay_unit='timestep')``, rather than the paper's
    Euler-residual form  z + dt*A*tanh(z)  (also supported, via ``PredictRes``
    on the same delayed edge; it performs on par here);
  - a soft, precision-weighted temporal prior (pi_p : pi_o = 5 : 1) relaxed
    jointly with the observation, rather than the paper's hard prior-reset
    followed by a single correction step;
  - K = 12 combined inference + learning (iPC) iterations per frame.
The accuracy gain comes from the latter two.

One-step prediction error lands ~2.9x below the paper's nonlinear model
(~0.028 vs ~0.082) and ~2x below the copy-last-frame persistence baseline;
5- and 10-step open-loop rollouts also beat their persistence counterparts.

Because the weights change every frame, h-step forecasts need the weights *as
they were at frame t*; these are recovered by replaying the (closed-form)
weight updates from the logged latent trajectory, checked against the final
trained weights.

No dataset needed. Saves a phase portrait (PNG) and a two-link-linkage
animation (GIF) next to this script. Runs in ~1 min; this model is tiny, so
the CPU backend is fastest: ``JAX_PLATFORMS=cpu uv run python
examples/temporal_pendulum.py``.
"""
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from matplotlib import animation

import pcn

SEED = 0
DT = 0.1                    # frame interval (s) = ODE step
T = 25004                   # frames (the paper runs 2500.4 s)
NOISE_STD = 0.1
K = 12                      # inference+learning iterations per frame
PI_P, PI_O = 5.0, 1.0       # transition / observation precision (soft prior)
LR_VALUES = 0.05            # relaxation step size
LR_W = DT * 0.9 / PI_P      # transition lr  (paper's k2, mapped)
LR_C0 = DT * 8.5 / 2.0      # observation lr (paper's k1, mapped, halved)
HORIZONS = (1, 5, 10)
OUT_DIR = Path(__file__).parent


def simulate_pendulum(rng):
    """RK4 pendulum (g=9.81, L=3), returns clean state and noisy obs (2, T)."""
    def f(th):
        return np.array([th[1], -(9.81 / 3.0) * np.sin(th[0])])

    clean = np.zeros((2, T))
    th = np.array([1.8, 2.2])
    for t in range(T):
        clean[:, t] = th
        k1, k2 = f(th), f(th + 0.5 * DT * f(th))
        k3 = f(th + 0.5 * DT * k2)
        k4 = f(th + DT * k3)
        th = th + DT / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    obs = clean + rng.normal(0, NOISE_STD, clean.shape)
    return clean, obs.astype(np.float32)


def build_net():
    net = pcn.PCNetwork(seed=SEED)
    eye = jnp.eye(2, dtype=jnp.float32)
    with net:
        l_x = pcn.Layer(dim=2, activation=pcn.Direct(), label='obs')
        l_z = pcn.Layer(dim=2, activation=pcn.Tanh(), label='z')
        # init_precision = 2 * D * pi: D cancels the backend's mean-over-dims
        # convention, the 2 cancels pre_scales (z is the pre of both edges).
        pcn.Predict(l_z, l_z, delay=1, delay_unit='timestep',
                    init_weight=eye, learn_weights=True,
                    init_precision=4 * PI_P, learn_precision=False,
                    use_bias=False, label='transition')
        pcn.Predict(l_z, l_x, init_weight=eye, learn_weights=True,
                    init_precision=4 * PI_O, learn_precision=False,
                    use_bias=False, label='observation')
    net.build()
    return net, l_x, l_z


def lr_c_schedule():
    """The paper's k1 decay: /1.015 every 300 frames (= 300*K optimizer steps)."""
    return optax.exponential_decay(LR_C0, transition_steps=300 * K,
                                   decay_rate=1 / 1.015, staircase=True)


def train_online(net, l_x, l_z, obs):
    """Single online pass: one combined value+weight step per iteration,
    the clamped frame advancing every K iterations."""
    popt = net.multi_transform(
        {'transition': optax.sgd(LR_W), 'observation': optax.sgd(lr_c_schedule())},
        default_optim=optax.set_to_zero())
    sim = pcn.Simulation(net)
    sim.train([{'obs': jnp.asarray(obs.T[None])}], data_map={l_x: 'obs'},
              epochs=1, iterations_per_sample=0,
              learning_iterations_per_sample=T * K,
              log_every=1, save_logs=True, convergence_threshold=0.0,
              values_optimizer=optax.sgd(LR_VALUES), params_optimizer=popt,
              feedforward_init=False, verbose=False)
    z_log = np.asarray(sim.logs['values'][0][l_z._idx])[:, 0, :]  # (T*K, 2)
    labels = [c.label for c in net.structure.predict_conns]
    w_final = np.asarray(sim.params.predict_weights[labels.index('transition')])
    c_final = np.asarray(sim.params.predict_weights[labels.index('observation')])
    return z_log, w_final, c_final


def replay_weights(obs, z_log):
    """Recover the per-frame weight trajectory from the logged latents.

    Each update is closed-form in the logged values (gradients are taken at
    the carry-in state of every iteration), so walking the log reproduces the
    exact weight sequence; the endpoint is asserted against the trained net.
    """
    lr_c = lr_c_schedule()
    W, C = np.eye(2), np.eye(2)
    Ws, Cs = np.zeros((T, 2, 2)), np.zeros((T, 2, 2))
    for i in range(T * K):
        t = i // K
        if i % K == 0:
            Ws[t], Cs[t] = W, C
        z_in = z_log[i - 1] if i > 0 else np.zeros(2)
        latch = z_log[t * K - 1] if t > 0 else np.zeros(2)  # z at frame t-1
        W = W + LR_W * PI_P * np.outer(z_in - W @ np.tanh(latch), np.tanh(latch))
        C = C + float(lr_c(i)) * PI_O * np.outer(obs[:, t] - C @ np.tanh(z_in),
                                                 np.tanh(z_in))
    return Ws, Cs, W, C


def forecast(Ws, Cs, z_frames, h):
    """h-step open-loop forecast: transition rolled h times from z[t-h].

    Weights are frozen at the anchor (post-update of frame t-h): a true
    forecast may not use W updated on frames inside the look-ahead gap.
    """
    preds = np.full((2, T), np.nan)
    for t in range(h, T):
        z = z_frames[t - h]
        Wa, Ca = Ws[min(t - h + 1, T - 1)], Cs[min(t - h + 1, T - 1)]
        for _ in range(h):
            z = Wa @ np.tanh(z)
        preds[:, t] = Ca @ np.tanh(z)
    return preds


def mse(a, b):
    m = ~np.isnan(a[0])
    return float(((a[:, m] - b[:, m]) ** 2).mean())


def linkage(th1, th2):
    """Two-link chain: link 1 at angle theta_1, link 2 at theta_2 (velocity
    drawn as an angle so both state dims are visible)."""
    p1 = np.array([np.sin(th1), -np.cos(th1)])
    p2 = p1 + np.array([np.sin(th2), -np.cos(th2)])
    return p1, p2


def draw_linkage(ax, th1, th2, color, filled):
    p1, p2 = linkage(th1, th2)
    style = (dict(color=color, lw=7, solid_capstyle='round') if filled
             else dict(color=color, lw=2))
    ax.plot([0, p1[0]], [0, p1[1]], zorder=3, **style)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], zorder=3, **style)
    ax.add_patch(plt.Circle(p1, 0.075, facecolor=color if filled else 'none',
                            edgecolor=color, lw=2, zorder=4))


def main():
    rng = np.random.default_rng(SEED)
    clean, obs = simulate_pendulum(rng)

    net, l_x, l_z = build_net()
    print(f"training online: T={T} frames, K={K} iterations/frame ...")
    z_log, w_final, c_final = train_online(net, l_x, l_z, obs)

    Ws, Cs, w_replay, c_replay = replay_weights(obs, z_log)
    dev = max(np.abs(w_replay - w_final).max(), np.abs(c_replay - c_final).max())
    assert dev < 1e-4, f"weight replay diverged from trained net ({dev:.2e})"
    z_frames = z_log[K - 1::K]                  # relaxed posterior per frame

    print(f"\nfull-run prediction MSE (noise floor = sigma^2 = "
          f"{NOISE_STD**2:.4f}):")
    print(f"{'horizon':>8} | {'tPC':>8} | {'persistence':>11}")
    preds = {}
    for h in HORIZONS:
        preds[h] = forecast(Ws, Cs, z_frames, h)
        persist = np.full((2, T), np.nan)
        persist[:, h:] = obs[:, :-h]
        print(f"{h:>8} | {mse(preds[h], obs):8.4f} | {mse(persist, obs):11.4f}")

    # Phase portrait, last 80 frames (the paper's Fig. 7 panel).
    lo = T - 80
    X, Y = np.mgrid[-np.pi:np.pi:30j, -4:4:30j]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.quiver(X, Y, Y, -(9.81 / 3.0) * np.sin(X), color='0.75', width=0.003)
    ax.plot(obs[0, lo:], obs[1, lo:], 'k-', lw=3, label='observed (noisy)')
    ax.plot(preds[1][0, lo:], preds[1][1, lo:], color='#E69F00', lw=2.2,
            label='tPC 1-step prediction')
    ax.set_xlabel(r'$\theta_1$'), ax.set_ylabel(r'$\theta_2$')
    ax.set_title('Pendulum phase portrait, last 80 frames')
    ax.legend(fontsize=9)
    fig.savefig(OUT_DIR / 'temporal_pendulum_phase.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    # Animation: filled linkage = true state, outlines = 1- and 5-step forecasts.
    frames = np.arange(T - 250, T)
    fig, ax = plt.subplots(figsize=(5.2, 5.4))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.02)

    def draw(t):
        ax.clear()
        draw_linkage(ax, clean[0, t], clean[1, t], '0.15', filled=True)
        draw_linkage(ax, preds[1][0, t], preds[1][1, t], '#E69F00', filled=False)
        draw_linkage(ax, preds[5][0, t], preds[5][1, t], '#56B4E9', filled=False)
        ax.legend(handles=[
            plt.Line2D([], [], color='0.15', lw=6, label='true'),
            plt.Line2D([], [], color='#E69F00', lw=2, label='predicted, 1 back'),
            plt.Line2D([], [], color='#56B4E9', lw=2, label='predicted, 5 back'),
        ], loc='upper left', fontsize=8)
        ax.set_title('Temporal PC pendulum tracking\n'
                     r'link 1 = $\theta_1$, link 2 = $\theta_2$ (velocity dim)'
                     f'   t = {t * DT:.1f} s', fontsize=10)
        ax.set_xlim(-2.35, 2.35), ax.set_ylim(-2.35, 2.35)
        ax.set_aspect('equal'), ax.axis('off')

    anim = animation.FuncAnimation(fig, lambda i: draw(frames[i]),
                                   frames=len(frames))
    anim.save(OUT_DIR / 'temporal_pendulum.gif',
              writer=animation.PillowWriter(fps=20), dpi=90)
    plt.close(fig)
    print(f"\nsaved {OUT_DIR / 'temporal_pendulum_phase.png'}")
    print(f"saved {OUT_DIR / 'temporal_pendulum.gif'}")


if __name__ == '__main__':
    main()
