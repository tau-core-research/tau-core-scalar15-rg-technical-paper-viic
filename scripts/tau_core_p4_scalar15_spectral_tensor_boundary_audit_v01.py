#!/usr/bin/env python3
"""Audit a common spectral-tensor origin for the scalar-15 boundary ray."""

import json
from pathlib import Path

import numpy as np


RNG = np.random.default_rng(20260810)
N = 5


def rc(shape):
    return RNG.normal(size=shape) + 1j * RNG.normal(size=shape)


def su5():
    q, r = np.linalg.qr(rc((N, N)))
    p = np.diag(r)
    q = q @ np.diag(np.conj(p) / np.abs(p))
    return q / np.linalg.det(q) ** (1 / N)


def fields():
    h = rc((N, 1))
    x = rc((N, N))
    delta = (x + x.T) / 2
    x = rc((N, N))
    sigma = (x + x.conj().T) / 2
    sigma -= np.trace(sigma) * np.eye(N) / N
    return h, delta, sigma


def operator(h, delta, sigma):
    return np.block(
        [
            [np.zeros((1, 1)), h.conj().T, h.T],
            [h, sigma, delta],
            [h.conj(), delta.conj().T, sigma.T],
        ]
    )


def cubic_basis(h, delta, sigma):
    rho = sigma @ delta + delta @ sigma.T
    return np.array(
        [
            (h.conj().T @ sigma @ h).item().real,
            np.real((h.conj().T @ delta @ h.conj()).item()),
            np.trace(delta.conj().T @ rho).real,
            np.trace(sigma @ sigma @ sigma).real,
        ]
    )


def quartic_basis(h, delta):
    dd = delta @ delta.conj().T
    nh = (h.conj().T @ h).item().real
    nd = np.trace(dd).real
    return np.array(
        [
            nh**2,
            nh * nd,
            (h.conj().T @ dd @ h).item().real,
            nd**2,
            np.trace(dd @ dd).real,
        ]
    )


cubic_rows, cubic_targets = [], []
quartic_rows, quartic_targets = [], []
max_covariance_error = 0.0
for _ in range(80):
    h, delta, sigma = fields()
    d = operator(h, delta, sigma)
    u = su5()
    w = np.block(
        [
            [np.ones((1, 1)), np.zeros((1, N)), np.zeros((1, N))],
            [np.zeros((N, 1)), u, np.zeros((N, N))],
            [np.zeros((N, 1)), np.zeros((N, N)), u.conj()],
        ]
    )
    d_transformed = operator(u @ h, u @ delta @ u.T, u @ sigma @ u.conj().T)
    max_covariance_error = max(
        max_covariance_error,
        float(np.max(np.abs(d_transformed - w @ d @ w.conj().T))),
    )
    cubic_rows.append(cubic_basis(h, delta, sigma))
    cubic_targets.append(np.trace(np.linalg.matrix_power(d, 3)).real)

    zero = np.zeros_like(sigma)
    d0 = operator(h, delta, zero)
    tr2 = np.trace(d0 @ d0).real
    quartic_rows.append(quartic_basis(h, delta))
    quartic_targets.append(np.trace(np.linalg.matrix_power(d0, 4)).real + tr2**2)

cubic_rows = np.vstack(cubic_rows)
quartic_rows = np.vstack(quartic_rows)
cubic_targets = np.array(cubic_targets)
quartic_targets = np.array(quartic_targets)
cubic_coeff, *_ = np.linalg.lstsq(cubic_rows, cubic_targets, rcond=None)
quartic_coeff, *_ = np.linalg.lstsq(quartic_rows, quartic_targets, rcond=None)
cubic_error = float(np.linalg.norm(cubic_rows @ cubic_coeff - cubic_targets) / np.linalg.norm(cubic_targets))
quartic_error = float(np.linalg.norm(quartic_rows @ quartic_coeff - quartic_targets) / np.linalg.norm(quartic_targets))

expected_cubic = np.array([6.0, 6.0, 3.0, 2.0])
expected_quartic = np.array([24.0, 16.0, 8.0, 4.0, 2.0])
checks = {
    "operator_is_gauge_covariant": max_covariance_error < 1e-10,
    "cubic_trace_identity_is_exact": cubic_error < 1e-12,
    "cubic_coefficients_match_6_6_3_2": np.allclose(cubic_coeff, expected_cubic, atol=1e-10),
    "spectral_tensor_quartic_identity_is_exact": quartic_error < 1e-12,
    "quartic_coefficients_match_24_16_8_4_2": np.allclose(quartic_coeff, expected_quartic, atol=1e-10),
    "both_portals_are_nonzero": bool(quartic_coeff[1] > 0 and quartic_coeff[2] > 0),
    "both_scalar15_quartics_are_nonzero": bool(quartic_coeff[3] > 0 and quartic_coeff[4] > 0),
    "breaking_to_cubic_ratio_is_one_half": abs(cubic_coeff[2] / cubic_coeff[1] - 0.5) < 1e-12,
}

report = {
    "audit": "P4-NU15SPEC-C1/N1",
    "status": "PASS" if all(checks.values()) else "FAIL",
    "carrier": "C direct-sum C^5 direct-sum conjugate(C^5)",
    "cubic_basis_order": ["Hdag Sigma H", "Re(Hdag Delta H*)", "Delta-adjoint breaking", "Tr Sigma^3"],
    "cubic_coefficients": cubic_coeff.tolist(),
    "quartic_basis_order": ["(Hdag H)^2", "portal_1", "portal_2", "quartic_1", "quartic_2"],
    "quartic_coefficients": quartic_coeff.tolist(),
    "max_covariance_error": max_covariance_error,
    "relative_cubic_fit_error": cubic_error,
    "relative_quartic_fit_error": quartic_error,
    "checks": {key: bool(value) for key, value in checks.items()},
    "claim_boundary": (
        "The declared spectral-tensor completion selects a nonzero scalar-15 "
        "boundary ray and fixed relative coefficients. Its occupation, the relative "
        "weights of the cubic and quartic spectral moments, the source scale, and the "
        "full beta functions are not yet derived from the physical base-seed parent."
    ),
}

out = Path(__file__).with_suffix(".json")
out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
