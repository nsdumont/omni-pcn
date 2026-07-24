"""
Learning rule classes for Project and Modulate connections.

These define how weights are updated for non-PC connections. Each rule has a
type_id used to select the update computation in the backend.
"""

from typing import Callable, Any


class LearningRule:
    """
    Base class for learning rules.

    Subclasses must define:
        type_id: int - identifier for backend switch statements
        learning_rate: float - learning rate for weight updates
    """
    type_id: int = 0
    learning_rate: float = 1e-3

    def __repr__(self):
        return f"{self.__class__.__name__}(learning_rate={self.learning_rate})"


class Hebbian(LearningRule):
    """
    Standard Hebbian learning rule.

    Weight update: dW = learning_rate * post @ pre.T

    Uses only pre and post activations - no external signal needed.

    Args:
        learning_rate: Learning rate for weight updates (default: 1e-3)
    """
    type_id = 0

    def __init__(self, learning_rate: float = 1e-3):
        self.learning_rate = learning_rate


class ThreeFactorHebbian(LearningRule):
    """
    Three-factor Hebbian learning with reward modulation.

    Weight update: dW = learning_rate * reward * post @ pre.T

    Uses pre, post, and a reward signal derived from NodeRefs and/or
    sample-dict keys. The signature matches ``GradientDescent.loss_fn``:

        reward_fn = ((inputs,), fn)

    where ``inputs`` is a NodeRef, a string (sample dict key), or a tuple
    mixing both, and ``fn`` receives the resolved arrays as positional
    args and returns a ``(batch,)`` or scalar reward.

    Args:
        learning_rate: Learning rate for weight updates (default: 1e-3)
        reward_fn: Optional ``(inputs, fn)`` tuple.

    Example::

        # Negative MSE reward: reads l_out.value and the 'label' sample key
        def neg_mse(pred, label):
            return -jnp.mean((pred - label) ** 2, axis=-1)
        rule = ThreeFactorHebbian(
            learning_rate=1e-4,
            reward_fn=((l_out.value, 'label'), neg_mse),
        )
    """
    type_id = 1

    def __init__(self, learning_rate: float = 1e-3, reward_fn=None):
        self.learning_rate = learning_rate
        if reward_fn is not None:
            if not (isinstance(reward_fn, tuple) and len(reward_fn) == 2):
                raise ValueError(
                    "reward_fn must be an (inputs, fn) tuple, e.g. "
                    "((node_ref, 'label'), my_fn). Got: " + repr(reward_fn))
            self.reward_fn_inputs, self.reward_fn_callable = reward_fn
        else:
            self.reward_fn_inputs = None
            self.reward_fn_callable = None

    @property
    def reward_fn(self):
        """Return the (inputs, fn) tuple, or None."""
        if self.reward_fn_callable is not None:
            return (self.reward_fn_inputs, self.reward_fn_callable)
        return None


class Oja(LearningRule):
    """
    Oja's learning rule — Hebbian with weight normalization.

    Weight update: dW = learning_rate * (post @ pre.T - post^2 * W)

    The subtracted term prevents runaway weight growth by introducing
    activity-dependent decay. The weight vector converges toward the first
    principal component of the input with unit norm.

    Args:
        learning_rate: Learning rate for weight updates (default: 1e-3)
    """
    type_id = 3

    def __init__(self, learning_rate: float = 1e-3):
        self.learning_rate = learning_rate


class NoLearning(LearningRule):
    """
    No weight updates. For fixed connections (e.g., identity/residual shortcuts).

    Weights initialized with init_weight on Project/Modulate remain frozen.

    type_id = -1 (not matched by any learning branch in the backend).
    """
    type_id = -1

    def __init__(self):
        self.learning_rate = 0.0


class GradientDescent(LearningRule):
    """
    Gradient descent on a custom loss.

    ``loss_fn`` is **required**: it specifies which arrays to pass to the loss
    function, using the same ``(inputs, fn)`` pattern as ``record_map``. The
    connection weight is updated by a single-step ``jax.grad`` of the loss.

    .. note::
        The legacy "energy" path (``GradientDescent()`` with no ``loss_fn``,
        which learned the weight from the PC energy backward pass) has been
        removed. Project/Modulate weights are now either learned from an
        explicit ``loss_fn`` here, or via :class:`Hebbian` / :class:`Oja` /
        :class:`ThreeFactorHebbian`. PC energy descent still drives **value**
        inference and **Predict** weight learning as before.

    Learning rate is controlled by the params_optimizer passed to
    ``Simulation.train()`` (or ``run_batch``). Use ``net.multi_transform()``
    to set per-connection learning rates.

    Args:
        loss_fn: Required ``(inputs, fn)`` tuple.
            ``inputs`` is a NodeRef, str (sample dict key), or tuple mixing
            both.  ``fn`` receives the resolved arrays as positional args and
            must return a scalar loss.

    Example::

        # Cross-entropy loss using output error and labels from the sample
        loss = GradientDescent(
            loss_fn=((p_out.error, 'label'), my_ce_loss),
        )
        Project(pre, post, update_rule=loss)
    """
    type_id = 2

    def __init__(self, loss_fn=None):
        self.learning_rate = 0.0
        if loss_fn is None:
            raise TypeError(
                "GradientDescent requires a loss_fn: an (inputs, fn) tuple, "
                "e.g. ((node_ref, 'label'), my_fn). The energy-based path "
                "(no loss_fn) has been removed; use Hebbian, Oja, or "
                "ThreeFactorHebbian to learn Project/Modulate weights without "
                "an explicit loss.")
        if not (isinstance(loss_fn, tuple) and len(loss_fn) == 2):
            raise ValueError(
                "loss_fn must be a (inputs, fn) tuple, e.g. "
                "((node_ref, 'label'), my_fn). Got: " + repr(loss_fn))
        self.loss_fn_inputs, self.loss_fn_callable = loss_fn

    @property
    def loss_fn(self):
        """Return the (inputs, fn) tuple."""
        return (self.loss_fn_inputs, self.loss_fn_callable)
