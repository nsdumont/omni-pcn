"""Hierarchical temporal PC: learning the flow map of a chaotic double pendulum.

A double pendulum with slightly unequal arms (L1=1.0, L2=0.9) is chaotic. A
two-level hierarchical temporal PCN — delayed self-edges at both latent levels
plus a top-down edge:

    Predict(z2, z2, delay=1, delay_unit='timestep')   # slow recurrence
    Predict(z2, z1)                                   # top-down context
    Predict(z1, z1, delay=1, delay_unit='timestep')   # fast recurrence
    Predict(z1, x)                                    # readout

is trained ONLINE (combined inference+learning every iteration) on a batch of
64 trajectories with random initial conditions, 30 passes, so it must learn
the global dynamics, not one orbit. A strong transition prior (pi_p : pi_o =
10 : 1) trades a little anchoring accuracy for much better rollouts.

Evaluation — the interesting part — uses PERIODIC CLAMPING on 8 held-out
trajectories with frozen weights: the state is provided only every X=12
frames (3-frame-wide clamps via a temporal mask in the data_map), and the
BATCH dimension phase-shifts the clamp grid, so a single batched ``sim.test``
scores every frame at every look-ahead age 1..9 since its last observation.
Printed against the copy-last-observation persistence baseline; look-ahead
error stays ~3-7x below it across the whole range.

Observations are noiseless: with this recipe, observation noise was measured
to be a minor term — the curve isolates dynamics-model error. Saves a
look-ahead curve (PNG) and a two-bar-linkage animation (GIF) next to this
script. Runs in a few minutes; CPU is fastest at this size:
``JAX_PLATFORMS=cpu uv run python examples/temporal_double_pendulum.py``.
"""
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from matplotlib import animation

import pcn

SEED = 0
G, L1, L2 = 9.81, 1.0, 0.9              # slightly unequal arms -> chaotic
DT, SUB = 0.05, 4                       # frame step; RK4 substeps per frame
N_TRAIN, N_TEST = 64, 8
T_TRAIN, T_TEST = 2000, 1200
TH_MAX = 2.6                            # reject near-flip trajectories

Z_DIM = 32
PI_P, PI_TD, PI_O = 10.0, 1.0, 1.0      # transition / top-down / obs precision
K, EPOCHS = 12, 30                      # iterations per frame; training passes
LR_VALUES, LR_TRANS = 0.05, 0.009
LR_OBS0 = 0.425                         # decays /1.015 every 300 frames
EPS = 0.05                              # init noise (exact identities leave
                                        # extra latent dims with zero gradient)
X, CLAMP_W = 12, 3                      # eval: clamp period and width
K_EVAL, LR_X_EVAL, LR_Z_EVAL = 48, 0.4, 0.01
BURN = 3 * X
OUT_DIR = Path(__file__).parent


# ----------------------------- the system ------------------------------------
def deriv(s):
    th1, th2, w1, w2 = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    d = th1 - th2
    den = 3.0 - np.cos(2 * d)           # 2*m1 + m2 - m2*cos(2d), m1=m2=1
    dw1 = (-3 * G * np.sin(th1) - G * np.sin(th1 - 2 * th2)
           - 2 * np.sin(d) * (w2 ** 2 * L2 + w1 ** 2 * L1 * np.cos(d))
           ) / (L1 * den)
    dw2 = (2 * np.sin(d) * (w1 ** 2 * L1 * 2 + 2 * G * np.cos(th1)
                            + w2 ** 2 * L2 * np.cos(d))) / (L2 * den)
    return np.stack([w1, w2, dw1, dw2], axis=-1)


def simulate(s0, T):
    out = np.zeros((s0.shape[0], T, 4))
    s = s0.copy()
    for t in range(T):
        out[:, t] = s
        for _ in range(SUB):
            k1 = deriv(s)
            k2 = deriv(s + 0.5 * DT / SUB * k1)
            k3 = deriv(s + 0.5 * DT / SUB * k2)
            k4 = deriv(s + DT / SUB * k3)
            s = s + DT / SUB / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return out


def sample_trajectories(rng, n, T):
    keep = []
    while len(keep) < n:
        m = 3 * (n - len(keep))
        s0 = np.stack([rng.uniform(-1.6, 1.6, m), rng.uniform(-1.6, 1.6, m),
                       rng.uniform(-1.0, 1.0, m), rng.uniform(-1.0, 1.0, m)],
                      axis=-1)
        tr = simulate(s0, T)
        keep.extend(tr[np.abs(tr[:, :, :2]).max(axis=(1, 2)) < TH_MAX])
    return np.stack(keep[:n]).astype(np.float32)       # (n, T, 4)


# ----------------------------- the model -------------------------------------
def build_net():
    net = pcn.PCNetwork(seed=SEED)
    rng = np.random.default_rng(SEED + 2)

    def init(shape, base=None):
        w = EPS * rng.standard_normal(shape)
        return jnp.asarray(w if base is None else w + base, jnp.float32)

    with net:
        l_x = pcn.Layer(dim=4, activation=pcn.Direct(), label='obs')
        l_z = pcn.Layer(dim=Z_DIM, activation=pcn.Tanh(), label='z')
        l_z2 = pcn.Layer(dim=Z_DIM, activation=pcn.Tanh(), label='z2')
        # init_precision = 2 * D_post * pi: D_post cancels the backend's
        # mean-over-dims; the 2 cancels pre_scales (each latent is the pre of
        # two Predict connections).
        pcn.Predict(l_z, l_z, delay=1, delay_unit='timestep',
                    init_weight=init((Z_DIM, Z_DIM), np.eye(Z_DIM)),
                    learn_weights=True, init_precision=2 * Z_DIM * PI_P,
                    learn_precision=False, use_bias=False, label='transition')
        pcn.Predict(l_z, l_x, init_weight=init((4, Z_DIM), np.eye(4, Z_DIM)),
                    learn_weights=True, init_precision=2 * 4 * PI_O,
                    learn_precision=False, use_bias=False, label='observation')
        pcn.Predict(l_z2, l_z2, delay=1, delay_unit='timestep',
                    init_weight=init((Z_DIM, Z_DIM), np.eye(Z_DIM)),
                    learn_weights=True, init_precision=2 * Z_DIM * PI_P,
                    learn_precision=False, use_bias=False, label='transition2')
        pcn.Predict(l_z2, l_z, init_weight=init((Z_DIM, Z_DIM), np.eye(Z_DIM)),
                    learn_weights=True, init_precision=2 * Z_DIM * PI_TD,
                    learn_precision=False, use_bias=False, label='topdown')
    net.build()
    return net, l_x


def train(net, l_x, trajs):
    lr_obs = optax.exponential_decay(LR_OBS0, transition_steps=300 * K,
                                     decay_rate=1 / 1.015, staircase=True)
    popt = net.multi_transform(
        {'transition': optax.sgd(LR_TRANS), 'transition2': optax.sgd(LR_TRANS),
         'topdown': optax.sgd(LR_TRANS), 'observation': optax.sgd(lr_obs)},
        default_optim=optax.set_to_zero())
    sim = pcn.Simulation(net)
    sim.train([{'obs': jnp.asarray(trajs)}], data_map={l_x: 'obs'},
              epochs=EPOCHS, iterations_per_sample=0,
              learning_iterations_per_sample=T_TRAIN * K,
              log_every=T_TRAIN * K, convergence_threshold=0.0,
              values_optimizer=optax.sgd(LR_VALUES), params_optimizer=popt,
              feedforward_init=False, verbose=False)
    return sim


def lookahead_eval(sim, l_x, trajs):
    """Periodic clamping, batch-phase-shifted: element (i, b) clamps frames
    with (t - b) % X < CLAMP_W, so every frame of every trajectory is scored
    at every age since its last observation. The eval values optimizer is
    per-layer: the released obs node needs a fast step (0.4), the latents a
    gentle one (0.01 — the learned readout amplifies their correction)."""
    n = trajs.shape[0]
    t_idx = np.arange(T_TEST)
    data = np.repeat(trajs, X, axis=0)
    mask = np.zeros((n * X, T_TEST, 4), np.float32)
    for b in range(X):
        mask[b::X, ((t_idx - b) % X) < CLAMP_W, :] = 1.0
    labels = tuple('x' if i == l_x._idx else 'z'
                   for i in range(len(sim.net._layers)))
    vopt = optax.multi_transform(
        {'x': optax.sgd(LR_X_EVAL), 'z': optax.sgd(LR_Z_EVAL)}, labels)
    res = sim.test([{'obs': jnp.asarray(data), 'mask': jnp.asarray(mask)}],
                   data_map={l_x: ('obs', 'mask')},
                   iterations_per_sample=T_TEST * K_EVAL,
                   feedforward_init=False, values_optimizer=vopt,
                   log_every=K_EVAL, verbose=False, return_logs=True)
    x_gen = np.asarray(res['values'][l_x._idx])[:, -T_TEST:, :]
    streams = np.zeros((X - CLAMP_W, n, T_TEST, 4))
    for a in range(1, X - CLAMP_W + 1):
        for t in range(T_TEST):
            streams[a - 1, :, t] = x_gen[(t - a - (CLAMP_W - 1)) % X::X, t]
    mse = ((streams - trajs[None]) ** 2)[:, :, BURN:].mean(axis=(1, 2, 3))
    return streams, mse


def linkage(th1, th2):
    p1 = L1 * np.array([np.sin(th1), -np.cos(th1)])
    return p1, p1 + L2 * np.array([np.sin(th2), -np.cos(th2)])


def draw_linkage(ax, th1, th2, color, filled):
    p1, p2 = linkage(th1, th2)
    style = (dict(color=color, lw=7, solid_capstyle='round') if filled
             else dict(color=color, lw=2))
    ax.plot([0, p1[0]], [0, p1[1]], zorder=3, **style)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], zorder=3, **style)
    ax.add_patch(plt.Circle(p1, 0.06, facecolor=color if filled else 'none',
                            edgecolor=color, lw=2, zorder=4))


def main():
    rng = np.random.default_rng(SEED)
    print(f"simulating {N_TRAIN}+{N_TEST} double-pendulum trajectories ...")
    train_trajs = sample_trajectories(rng, N_TRAIN, T_TRAIN)
    test_trajs = sample_trajectories(rng, N_TEST, T_TEST)

    net, l_x = build_net()
    print(f"training online: {N_TRAIN} trajectories x {T_TRAIN} frames, "
          f"K={K}, {EPOCHS} passes ...")
    sim = train(net, l_x, train_trajs)

    print("evaluating with periodic clamping on held-out trajectories ...")
    streams, mse = lookahead_eval(sim, l_x, test_trajs)
    hs = np.arange(1, X - CLAMP_W + 1)
    persist = np.array([((test_trajs[:, BURN:] - test_trajs[:, BURN - h:-h])
                         ** 2).mean() for h in hs])
    print(f"\nheld-out look-ahead MSE (h = frames since last observation):")
    print(f"{'h':>3} | {'tPC':>8} | {'persistence':>11}")
    for h, m, p in zip(hs, mse, persist):
        print(f"{h:>3} | {m:8.4f} | {p:11.4f}")

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(hs, mse, 'o-', color='#E69F00', lw=2, label='hierarchical tPC')
    ax.plot(hs, persist, '--', color='0.4', lw=1.5, label='persistence')
    ax.set_yscale('log')
    ax.set_xlabel('look-ahead h (frames since last observation)')
    ax.set_ylabel('MSE (log)')
    ax.set_title('Held-out look-ahead, chaotic double pendulum')
    ax.grid(alpha=0.25), ax.legend(fontsize=9)
    fig.savefig(OUT_DIR / 'temporal_double_pendulum_lookahead.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    # GIF: median-difficulty held-out trajectory, ages 1 and 5.
    f0, n_frames = 100, 300
    win = ((streams[0, :, f0:f0 + n_frames] -
            test_trajs[:, f0:f0 + n_frames]) ** 2).mean(axis=(1, 2))
    traj = int(np.argsort(win)[len(win) // 2])
    truth = test_trajs[traj]
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.02)

    def draw(i):
        t = f0 + i
        ax.clear()
        draw_linkage(ax, truth[t, 0], truth[t, 1], '0.15', filled=True)
        draw_linkage(ax, *streams[0, traj, t, :2], '#E69F00', filled=False)
        draw_linkage(ax, *streams[4, traj, t, :2], '#56B4E9', filled=False)
        ax.legend(handles=[
            plt.Line2D([], [], color='0.15', lw=6, label='true'),
            plt.Line2D([], [], color='#E69F00', lw=2, label='predicted, 1 back'),
            plt.Line2D([], [], color='#56B4E9', lw=2, label='predicted, 5 back'),
        ], loc='upper left', fontsize=8)
        ax.set_title('Hierarchical tPC — held-out double pendulum\n'
                     f'state given every {X} frames   t = {t * DT:.2f} s',
                     fontsize=10)
        ax.set_xlim(-2.1, 2.1), ax.set_ylim(-2.1, 1.45)
        ax.set_aspect('equal'), ax.axis('off')

    anim = animation.FuncAnimation(fig, draw, frames=n_frames)
    anim.save(OUT_DIR / 'temporal_double_pendulum.gif',
              writer=animation.PillowWriter(fps=20), dpi=90)
    plt.close(fig)
    print(f"\nsaved {OUT_DIR / 'temporal_double_pendulum_lookahead.png'}")
    print(f"saved {OUT_DIR / 'temporal_double_pendulum.gif'}")


if __name__ == '__main__':
    main()
