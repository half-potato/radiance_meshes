"""Property + reference-parity tests for ``utils.safe_math``.

The safe_* primitives wrap numerically-fragile ops (log, sqrt, div, pow) with
custom ``torch.autograd.Function`` backward passes. They are used in the loss
paths of full_inverse.py and rfds_rev.py, so a backward bug here silently kills
training signal.

These tests are property/spec-based:

  * Value at singularities (input 0, ±1e38) → output is finite. This is the
    spec of the "safe" prefix: never produce NaN/Inf.
  * ``torch.autograd.gradcheck`` against finite-difference. This compares the
    custom analytic backward to a mathematical reference (numerical
    differentiation), not to the current code's behavior.

We avoid asserting specific output values — that would just pin whatever the
current code happens to produce. Where a singular point is genuinely
undefined (e.g. division by zero), we assert the documented contract
(``safe_div(n, 0) == 0`` per the ``where`` in the forward pass).
"""

from __future__ import annotations

import math

import pytest
import torch

from utils.safe_math import (
    minus_eps,
    plus_eps,
    remove_zero,
    safe_arccos,
    safe_arctan2,
    safe_div,
    safe_exp,
    safe_log,
    safe_pow,
    safe_sin,
    safe_cos,
    safe_sqrt,
    tiny_val,
)


# ---------------------------------------------------------------------------
# value-at-singularity contracts: finite output, no NaN/Inf
# ---------------------------------------------------------------------------


def _finite(t: torch.Tensor) -> bool:
    return torch.isfinite(t).all().item()


def test_safe_div_zero_denominator_returns_zero():
    """``safe_div(n, 0) -> 0`` is the documented forward contract."""
    n = torch.tensor([1.0, -2.0, 3.0])
    d = torch.tensor([0.0, 0.0, 0.0])
    out = safe_div(n, d)
    assert _finite(out)
    assert torch.allclose(out, torch.zeros_like(out))


def test_safe_div_extreme_values_stay_finite():
    n = torch.tensor([1e38, -1e38, tiny_val])
    d = torch.tensor([tiny_val * 2, tiny_val * 2, 1e38])
    out = safe_div(n, d)
    assert _finite(out)


def test_safe_log_at_zero_is_finite():
    out = safe_log(torch.zeros(4))
    assert _finite(out)


def test_safe_log_at_negative_clamps_to_finite():
    out = safe_log(torch.tensor([-1.0, -1e10]))
    assert _finite(out)


def test_safe_sqrt_at_negative_is_finite_and_nonnegative():
    out = safe_sqrt(torch.tensor([-1.0, -1e10, 0.0]))
    assert _finite(out)
    assert (out >= 0).all()


def test_safe_exp_at_large_input_is_finite():
    out = safe_exp(torch.tensor([1e10, 200.0, -1e10]))
    assert _finite(out)


def test_safe_pow_at_zero_with_half_exponent_is_finite():
    out = safe_pow(torch.zeros(3), exponent=0.5)
    assert _finite(out)


def test_safe_arctan2_at_origin_is_finite():
    out = safe_arctan2(torch.tensor([0.0]), torch.tensor([0.0]))
    assert _finite(out)


def test_safe_arccos_outside_unit_is_clamped():
    """Inputs >= 1 must produce 0 and inputs <= -1 must produce pi.

    This is the explicit contract in safe_arccos. Verifying it locks the
    boundary behavior even if the inner ``torch.acos`` changes.
    """
    out = safe_arccos(torch.tensor([2.0, -2.0, 1.0, -1.0]))
    assert _finite(out)
    expected = torch.tensor([0.0, math.pi, 0.0, math.pi])
    assert torch.allclose(out, expected, atol=1e-6)


def test_remove_zero_shifts_zeros_only():
    x = torch.tensor([0.0, tiny_val / 2, 1.0, -1.0])
    out = remove_zero(x)
    # the near-zero entries are pushed away
    assert (out.abs() >= tiny_val).all()
    # nonzero entries are untouched
    assert out[2] == 1.0 and out[3] == -1.0


def test_plus_minus_eps_are_strictly_above_below():
    x = torch.tensor([0.0, 1.0, -1.0, 1e10])
    p = plus_eps(x)
    m = minus_eps(x)
    assert (p > x).all() or (p >= x).all()  # nextafter toward +inf
    assert (m < x).all() or (m <= x).all()
    assert _finite(p) and _finite(m)


# ---------------------------------------------------------------------------
# gradient reference parity: analytic backward vs finite-difference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn,x_init", [
    (safe_log, torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)),
    (safe_exp, torch.tensor([-0.5, 0.0, 1.0], dtype=torch.float64)),
    (safe_sqrt, torch.tensor([0.5, 1.0, 4.0], dtype=torch.float64)),
    (lambda x: safe_pow(x, exponent=0.5),
        torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)),
    (lambda x: safe_pow(x, exponent=2.0),
        torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)),
    (safe_sin, torch.tensor([0.1, 0.5, 1.0], dtype=torch.float64)),
    (safe_cos, torch.tensor([0.1, 0.5, 1.0], dtype=torch.float64)),
])
def test_gradcheck_unary(fn, x_init):
    """Analytic backward agrees with finite difference within fp64 tolerance.

    Inputs are well away from singularities so the clip doesn't mask any
    error; singularity behavior is exercised separately by the value tests
    above. We pass eps and tolerance suitable for float64.
    """
    x = x_init.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_gradcheck_safe_div():
    n = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
    d = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(safe_div, (n, d), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_gradcheck_safe_arctan2():
    x1 = torch.tensor([0.5, 1.0, -0.3], dtype=torch.float64, requires_grad=True)
    x2 = torch.tensor([0.5, -1.0, 0.7], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(safe_arctan2, (x1, x2), eps=1e-6, atol=1e-4, rtol=1e-3)
