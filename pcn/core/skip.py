"""
Skip connection class for predictive coding networks.

Skip creates a delayed skip connection between two layers by inserting
auxiliary layers connected by fixed-weight Project connections. This is
useful for residual-style shortcuts across multiple hierarchy levels.
"""

from typing import TYPE_CHECKING, List, Optional

from .layer import Layer
from .connections import Project
from .learning_rules import NoLearning
from .activations import Direct

if TYPE_CHECKING:
    from .network import PCNetwork


class Skip:
    """
    Delayed skip connection between two equal-dim layers.

    Creates ``delay`` auxiliary layers (Direct activation) and ``delay + 1``
    Project connections forming a chain::

        pre -> aux_1 -> ... -> aux_delay -> post

    Each Project uses ``NoLearning`` with weight ``skip_scale * I``.

    This is equivalent to::

        l_aux = [pre]
        for i in range(delay):
            l_aux.append(Layer(dim=pre.dim, activation=Direct()))
        l_aux.append(post)
        for i in range(delay + 1):
            Project(l_aux[i].value, l_aux[i+1].value,
                    update_rule=NoLearning(),
                    init_weight=skip_scale * jnp.eye(l_aux[i].dim))

    Args:
        pre: Source layer.
        post: Target layer (must have same dim as pre).
        delay: Number of auxiliary layers inserted between pre and post.
        skip_scale: Scalar multiplier for the identity weight matrices.

    Raises:
        ValueError: If ``pre.dim != post.dim``.
        RuntimeError: If called outside a ``with net:`` block.

    Attributes:
        auxiliary_layers: List of the created intermediate layers.
        project_conns: List of the created Project connections.
    """

    def __init__(
        self,
        pre: Layer,
        post: Layer,
        delay: int = 1,
        skip_scale: float = 1.0,
    ):
        if pre.dim != post.dim:
            raise ValueError(
                f"Skip requires pre.dim == post.dim, "
                f"got pre.dim={pre.dim}, post.dim={post.dim}"
            )

        from .network import _get_current_network
        net = _get_current_network()

        self.pre = pre
        self.post = post
        self.delay = delay
        self.skip_scale = skip_scale

        self.auxiliary_layers, self.project_conns = net._add_skip(
            pre, post, delay, skip_scale
        )
