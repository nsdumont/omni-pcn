"""Bit-for-bit regression gate for the Phase-2 delay/hist carry refactor.

Phase 2 folds the dedicated ``prev_errors`` / ``prev_precisions`` one-step
inference carries into the ``hist`` ring-buffer structure as depth-1,
unit='iteration' buffers (dropped entirely when nothing reads them). This test
rebuilds the same 10-net consumer matrix used to capture ``phase2_golden.npz``
on the PRE-refactor code and asserts every logged array
(``values`` / ``errors`` / ``precisions`` / ``deltas``, test energies, and
post-train params) is ``np.array_equal`` (EXACT) to the golden.

The golden was captured deterministically (fixed-seed data, ``is_stochastic=
False`` so the key-free deterministic path is taken) and verified NaN-free +
reproducible across two passes by ``_golden_phase2_capture.main``.

Regenerate the golden (only on the last known-good code) with::

    python -m pcn.tests._golden_phase2_capture pcn/tests/phase2_golden.npz
"""

import os

import numpy as np
import pytest

from pcn.tests._golden_phase2_capture import collect_all

_GOLDEN = os.path.join(os.path.dirname(__file__), 'phase2_golden.npz')


@pytest.fixture(scope='module')
def golden():
    if not os.path.exists(_GOLDEN):
        pytest.skip(f"golden npz missing: {_GOLDEN}")
    return dict(np.load(_GOLDEN))


def test_phase2_bit_identical(golden):
    got = collect_all()

    gk, ck = set(golden), set(got)
    assert gk == ck, (
        f"key set changed: only-golden={sorted(gk - ck)[:8]} "
        f"only-new={sorted(ck - gk)[:8]}")

    mismatches = []
    for k in sorted(got):
        g = np.asarray(golden[k])
        c = np.asarray(got[k])
        if g.shape != c.shape or not np.array_equal(g, c):
            try:
                d = float(np.max(np.abs(
                    g.astype(np.float64) - c.astype(np.float64))))
            except Exception:
                d = float('nan')
            mismatches.append((k, g.shape, c.shape, d))

    assert not mismatches, (
        f"{len(mismatches)} non-identical arrays (showing 30):\n" + "\n".join(
            f"  {k}  gshape={gs} cshape={cs} max|Δ|={d}"
            for k, gs, cs, d in mismatches[:30]))
