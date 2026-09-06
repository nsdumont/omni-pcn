"""Sparse (CSR/CSC) weights for masked / banded linear transforms.

A :class:`SparseWeight` is a fixed sparsity structure plus a learnable
``data`` vector holding the nonzero entries of a ``(post_dim, pre_dim)``
weight matrix. It replaces the dense-matrix-times-mask representation of
``transformation='masked'`` / ``'banded{N}'`` connections when
``Predict/Project/Modulate(..., sparse=True | 'auto')`` is requested.

Design (see the workspace ``sparse_weights_redesign.md`` for the benchmarks):

* Only ``data`` is ever handed to optax. The backend swaps ``w.data`` into
  its ``trainable`` dict and rebuilds the ``SparseWeight`` after the update,
  so every optax transform (Adam, weight decay, ``multi_transform`` labels)
  works unchanged on a flat ``(nse,)`` leaf. The int32 structure arrays are
  ordinary pytree leaves — traced, donated, returned unchanged — so they are
  never baked into the HLO as constants (which XLA would constant-fold at
  compile time for seconds per million entries).
* The product ``x @ W.T`` is a :func:`jax.custom_vjp`:
  forward = cuSPARSE CSR SpMM; ``∂/∂x`` = CSR SpMM on the stored transposed
  structure (the CSC of ``W``); ``∂/∂data`` = a chunked row-gather einsum
  (``sampled_outer``). JAX's own transpose rule would route ``∂/∂x`` through
  a COO product that is ~25x slower, and its sampled-gradient primitive is
  ~10x slower than the chunked einsum (and OOMs when evaluated eagerly).
* cuSPARSE lowering is **off by default** in JAX; this module turns it on at
  import. Without it every sparse product lowers to a generic gather/scatter
  path that is slower than the dense product it replaces.

Memory per connection during a learn step: dense-masked ≈ 6 live
``(post, pre)`` f32 copies (W, mask, ``alpha·W.T`` temp, grad, Adam m, v);
sparse ≈ 16 B/nse (+ 8 B/nse Adam) + 4 B·(post + pre).
"""

from __future__ import annotations

import os
from typing import NamedTuple, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax, tree_util
from jax.experimental.sparse import BCSR, bcsr_dot_general

# cuSPARSE lowering of BCSR/BCOO products is OFF by default in JAX 0.9.x.
# Harmless on non-CUDA backends (only affects GPU lowering).
jax.config.update('jax_bcoo_cusparse_lowering', True)

#: Chunk size (in nonzeros) for the sampled weight gradient. Sweep on an RTX
#: 5090 (B=128): 2**16–2**18 is within 10% of the best at every nse tested;
#: a single-chunk map falls onto XLA's slow reduce fusion (8x cliff), so
#: :func:`sampled_outer` always forces at least two chunks.
SPARSE_CHUNK: int = 2 ** 18
#: ``sparse='auto'`` picks the sparse representation iff
#: ``density <= AUTO_MAX_DENSITY and post*pre >= AUTO_MIN_SIZE``.
#: Measured: speed win at <= ~2% density, break-even ~5%; below ~1024**2
#: the dense product is launch-bound and nothing matters.
AUTO_MAX_DENSITY: float = 0.05
AUTO_MIN_SIZE: int = 2 ** 20
#: jax-mps (Metal) has no ``custom_call`` handler, so cuSPARSE is impossible
#: there and the generic lowering is unverified. ``sparse=`` falls back to the
#: dense-masked path on Metal unless this is True (env ``PCN_SPARSE_ON_METAL=1``
#: or set ``pcn.core.sparse.SPARSE_ON_METAL = True`` before ``build()``).
SPARSE_ON_METAL: bool = os.environ.get('PCN_SPARSE_ON_METAL', '0') == '1'

_DN = (((1,), (0,)), ((), ()))   # W (post, pre) contracted with X (pre, B)


class SparseWeight(NamedTuple):
    """Fixed-structure sparse ``(post_dim, pre_dim)`` weight matrix.

    Attributes:
        data: ``(nse,)`` float32 — the nonzero values; the only learnable leaf.
        indices: ``(nse, 2)`` int32 COO ``(row, col)`` pairs, row-major sorted,
            unique. ``indices[:, 1]`` doubles as the CSR column-index array.
        indptr: ``(post_dim + 1,)`` int32 CSR row pointers.
        t_indptr: ``(pre_dim + 1,)`` int32 CSR row pointers of ``W.T`` (the
            CSC of ``W``), used for the backward product.
        t_perm: ``(nse,)`` int32 permutation: ``data[t_perm]`` is ``W.T``'s
            CSR data and ``indices[t_perm, 0]`` its column indices.
        shape: static ``(post_dim, pre_dim)``.
    """
    data: jnp.ndarray
    indices: jnp.ndarray
    indptr: jnp.ndarray
    t_indptr: jnp.ndarray
    t_perm: jnp.ndarray
    shape: tuple

    # ---- introspection -------------------------------------------------
    @property
    def nse(self) -> int:
        return int(self.indices.shape[0])

    @property
    def density(self) -> float:
        return self.nse / float(self.shape[0] * self.shape[1])

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def ndim(self) -> int:
        return 2

    def __repr__(self) -> str:
        return f"SparseWeight(shape={tuple(self.shape)}, nse={self.nse}, dtype={self.data.dtype})"

    # ---- conversions ---------------------------------------------------
    def with_data(self, data) -> 'SparseWeight':
        """Same structure, new ``data`` (shape ``(nse,)``)."""
        return self._replace(data=data)

    def todense(self) -> jnp.ndarray:
        """Materialise the dense ``(post_dim, pre_dim)`` matrix (explicit, never implicit)."""
        return jnp.zeros(tuple(self.shape), self.data.dtype).at[
            self.indices[:, 0], self.indices[:, 1]].set(self.data)

    def dense_mask(self) -> jnp.ndarray:
        """Binary ``(post_dim, pre_dim)`` float32 mask of the structure."""
        return jnp.zeros(tuple(self.shape), jnp.float32).at[
            self.indices[:, 0], self.indices[:, 1]].set(1.0)

    @classmethod
    def from_indices(cls, rows, cols, shape, data) -> 'SparseWeight':
        """Build from parallel ``rows``/``cols``/``data`` (any order, no duplicates)."""
        shape = _check_shape(shape)
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        cols = np.asarray(cols, dtype=np.int64).reshape(-1)
        data = np.asarray(data, dtype=np.float32).reshape(-1)
        if not (rows.shape == cols.shape == data.shape):
            raise ValueError(
                f"rows/cols/data must have the same length, got "
                f"{rows.shape}, {cols.shape}, {data.shape}")
        _check_ranges(rows, cols, shape)
        flat = rows * shape[1] + cols
        order = np.argsort(flat, kind='stable')
        flat = flat[order]
        if flat.size and np.any(flat[1:] == flat[:-1]):
            raise ValueError("SparseWeight.from_indices: duplicate (row, col) entries")
        return cls._build(rows[order], cols[order], shape, data[order])

    @classmethod
    def from_dense(cls, W, mask=None) -> 'SparseWeight':
        """Extract the nonzeros of ``W`` (or of ``W`` where ``mask != 0``)."""
        W = np.asarray(W, dtype=np.float32)
        shape = _check_shape(W.shape)
        if mask is None:
            rows, cols = np.nonzero(W)
        else:
            rows, cols = mask_to_indices(mask, shape)
        return cls._build(rows, cols, shape, W[rows, cols])

    @classmethod
    def _build(cls, rows, cols, shape, data) -> 'SparseWeight':
        """``rows``/``cols`` already row-major sorted and unique."""
        indptr, t_indptr, t_perm = build_csr_csc(rows, cols, shape)
        indices = np.stack([rows, cols], axis=1).astype(np.int32)
        return cls(
            data=jnp.asarray(data, dtype=jnp.float32),
            indices=jnp.asarray(indices),
            indptr=jnp.asarray(indptr),
            t_indptr=jnp.asarray(t_indptr),
            t_perm=jnp.asarray(t_perm),
            shape=shape,
        )


# Register with the structure arrays as (traced) children and ``shape`` as
# static aux. Overrides the default namedtuple flattening, which would have
# made the shape ints leaves.
tree_util.register_pytree_node(
    SparseWeight,
    lambda w: ((w.data, w.indices, w.indptr, w.t_indptr, w.t_perm), tuple(w.shape)),
    lambda shape, children: SparseWeight(*children, shape),
)


# ============================================================================
# Structure helpers (host-side numpy)
# ============================================================================

def _check_shape(shape) -> Tuple[int, int]:
    shape = tuple(int(s) for s in shape)
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError(f"SparseWeight shape must be 2-D with positive dims, got {shape}")
    return shape


def _check_ranges(rows, cols, shape) -> None:
    if rows.size and (rows.min() < 0 or rows.max() >= shape[0]
                      or cols.min() < 0 or cols.max() >= shape[1]):
        raise ValueError(f"sparse indices out of range for shape {shape}")


def build_csr_csc(rows, cols, shape):
    """CSR row pointers, CSC row pointers and the CSR->CSC data permutation.

    ``rows``/``cols`` must be row-major sorted and unique (as produced by
    :func:`mask_to_indices`).
    """
    post, pre = shape
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    indptr = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=post))]).astype(np.int32)
    t_perm = np.lexsort((rows, cols)).astype(np.int32)     # sort by (col, row)
    t_indptr = np.concatenate([[0], np.cumsum(np.bincount(cols, minlength=pre))]).astype(np.int32)
    return indptr, t_indptr, t_perm


def is_index_mask(mask) -> bool:
    """True for the sparse mask formats: ``(rows, cols)`` tuple, ``scipy.sparse``, ``BCOO``/``BCSR``."""
    if isinstance(mask, tuple) and len(mask) == 2:
        return True
    if hasattr(mask, 'tocoo'):                      # scipy.sparse
        return True
    if hasattr(mask, 'indices') and hasattr(mask, 'data') and hasattr(mask, 'shape'):
        return True                                 # jax.experimental.sparse BCOO / BCSR
    return False


def mask_to_indices(mask, shape) -> Tuple[np.ndarray, np.ndarray]:
    """Normalise any supported mask format to row-major sorted, unique ``(rows, cols)``.

    Accepted: a dense ``(post, pre)`` array (nonzeros are kept), a
    ``scipy.sparse`` matrix, a ``jax.experimental.sparse`` ``BCOO``/``BCSR``,
    or a ``(rows, cols)`` tuple of 1-D integer arrays. Duplicates are merged
    and out-of-range indices raise. Returns int64 numpy arrays.
    """
    shape = _check_shape(shape)
    if isinstance(mask, tuple) and len(mask) == 2:
        rows = np.asarray(mask[0]).reshape(-1)
        cols = np.asarray(mask[1]).reshape(-1)
        if rows.shape != cols.shape:
            raise ValueError("(rows, cols) mask: rows and cols must have the same length")
        if rows.size and not (np.issubdtype(rows.dtype, np.integer)
                              and np.issubdtype(cols.dtype, np.integer)):
            raise ValueError("(rows, cols) mask must contain integer indices")
        rows = rows.astype(np.int64); cols = cols.astype(np.int64)
    elif hasattr(mask, 'tocoo'):                    # scipy.sparse
        if tuple(mask.shape) != shape:
            raise ValueError(f"weight_mask shape {tuple(mask.shape)} does not match expected {shape}.")
        coo = mask.tocoo()
        keep = np.asarray(coo.data) != 0
        rows = np.asarray(coo.row)[keep].astype(np.int64)
        cols = np.asarray(coo.col)[keep].astype(np.int64)
    elif hasattr(mask, 'indices') and hasattr(mask, 'data') and hasattr(mask, 'shape'):
        if tuple(mask.shape) != shape:
            raise ValueError(f"weight_mask shape {tuple(mask.shape)} does not match expected {shape}.")
        bcoo = mask.to_bcoo() if hasattr(mask, 'to_bcoo') else mask
        idx = np.asarray(bcoo.indices)
        keep = np.asarray(bcoo.data) != 0
        rows = idx[keep, 0].astype(np.int64)
        cols = idx[keep, 1].astype(np.int64)
    else:
        dense = np.asarray(mask)
        if dense.shape != shape:
            raise ValueError(f"weight_mask shape {dense.shape} does not match expected {shape}.")
        rows, cols = np.nonzero(dense)
        rows = rows.astype(np.int64); cols = cols.astype(np.int64)
    _check_ranges(rows, cols, shape)
    flat = np.unique(rows * shape[1] + cols)        # sorted row-major, deduplicated
    return flat // shape[1], flat % shape[1]


def band_indices(m, n, n_bands) -> Tuple[np.ndarray, np.ndarray]:
    """Row-major sorted indices of ``|i - j| <= n_bands`` for an ``(m, n)`` matrix.

    Same set as ``PCNetwork._make_band_mask(m, n, n_bands)`` (absolute index
    distance to the main diagonal), built without the dense matrix.
    """
    i = np.arange(m, dtype=np.int64)
    lo = np.maximum(0, i - n_bands)
    hi = np.minimum(n - 1, i + n_bands)
    counts = np.maximum(0, hi - lo + 1)
    rows = np.repeat(i, counts)
    starts = np.repeat(lo, counts)
    offsets = np.arange(counts.sum(), dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
    return rows, starts + offsets


def indices_to_dense_mask(rows, cols, shape) -> np.ndarray:
    """Binary float32 ``(post, pre)`` mask from an index set (the dense fallback)."""
    mask = np.zeros(_check_shape(shape), dtype=np.float32)
    mask[np.asarray(rows), np.asarray(cols)] = 1.0
    return mask


def resolve_sparse_mode(mode, nse: int, shape) -> bool:
    """Turn ``sparse=True | 'auto'`` into a decision for a given structure."""
    if mode is True:
        return True
    if mode == 'auto':
        size = int(shape[0]) * int(shape[1])
        return size >= AUTO_MIN_SIZE and nse / size <= AUTO_MAX_DENSITY
    return False


# ============================================================================
# Backend helpers: split the learnable leaf out of a weight, and put it back
# ============================================================================

def learnable_leaf(w):
    """The optax-facing leaf of a weight: ``w.data`` for SparseWeight, else ``w``."""
    return w.data if isinstance(w, SparseWeight) else w


def rebuild_weight(original, leaf):
    """Inverse of :func:`learnable_leaf`: re-attach the structure of ``original``."""
    return original._replace(data=leaf) if isinstance(original, SparseWeight) else leaf


# ============================================================================
# Kernels
# ============================================================================

def _csr(data, cols, indptr, shape):
    return BCSR((data, cols, indptr), shape=tuple(shape),
                indices_sorted=True, unique_indices=True)


def sampled_outer(g, x, indices, chunk: int = SPARSE_CHUNK) -> jnp.ndarray:
    """``out[k] = sum_b g[b, indices[k, 0]] * x[b, indices[k, 1]]``.

    The batch-summed outer product ``g.T @ x`` sampled at an index set,
    without forming the dense ``(post, pre)`` product. Rows of ``g.T`` /
    ``x.T`` are gathered (contiguous) in chunks of ``chunk`` nonzeros under
    ``lax.map`` so the transient is ``O(chunk * B)`` regardless of nse.
    Always uses at least two chunks: a single-chunk map lowers onto a slow
    XLA reduce fusion (measured 8x slower).

    Args:
        g: ``(B, post)`` cotangent / post-synaptic array.
        x: ``(B, pre)`` pre-synaptic array.
        indices: ``(nse, 2)`` int32 ``(row, col)`` pairs.
        chunk: nonzeros per map step (static).
    """
    nse = indices.shape[0]
    chunk = max(1, min(int(chunk), -(-nse // 2)))
    n_chunks = -(-nse // chunk)
    pad = n_chunks * chunk - nse
    ic = jnp.pad(indices, ((0, pad), (0, 0))) if pad else indices
    ic = ic.reshape(n_chunks, chunk, 2)
    gT, xT = g.T, x.T

    def _chunk(c):
        return jnp.einsum('nb,nb->n', gT[c[:, 0]], xT[c[:, 1]])

    return lax.map(_chunk, ic).reshape(-1)[:nse]


@jax.custom_vjp
def sparse_matmul(w: SparseWeight, x: jnp.ndarray) -> jnp.ndarray:
    """``x @ W.T`` for a :class:`SparseWeight` ``W``: ``(B, pre) -> (B, post)``.

    Forward is a CSR SpMM (cuSPARSE on CUDA); the custom VJP computes
    ``∂/∂x`` with the stored CSC structure and ``∂/∂data`` with
    :func:`sampled_outer`. Both value- and weight-gradients may be taken in
    the same ``jax.grad`` call (as the PC energy backward pass does).
    """
    post, pre = w.shape
    W = _csr(w.data, w.indices[:, 1], w.indptr, (post, pre))
    return bcsr_dot_general(W, x.T, dimension_numbers=_DN).T


def _sparse_matmul_fwd(w, x):
    return sparse_matmul(w, x), (w, x)


def _sparse_matmul_bwd(res, g):
    w, x = res
    post, pre = w.shape
    WT = _csr(w.data[w.t_perm], w.indices[w.t_perm, 0], w.t_indptr, (pre, post))
    gx = bcsr_dot_general(WT, g.T, dimension_numbers=_DN).T          # (B, pre)
    gd = sampled_outer(g, x, w.indices)                               # (nse,)
    # Structure leaves are integer-valued: symbolic-zero cotangents.
    return SparseWeight(gd, None, None, None, None, w.shape), gx


sparse_matmul.defvjp(_sparse_matmul_fwd, _sparse_matmul_bwd)


__all__ = [
    'SparseWeight', 'sparse_matmul', 'sampled_outer',
    'mask_to_indices', 'band_indices', 'indices_to_dense_mask', 'is_index_mask',
    'build_csr_csc', 'resolve_sparse_mode', 'learnable_leaf', 'rebuild_weight',
    'SPARSE_CHUNK', 'AUTO_MAX_DENSITY', 'AUTO_MIN_SIZE', 'SPARSE_ON_METAL',
]
