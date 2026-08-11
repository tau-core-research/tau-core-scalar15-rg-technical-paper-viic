#!/usr/bin/env python3
"""Numerically audit the minimal SU(5)-invariant scalar-15 support family."""

import json
from pathlib import Path

import numpy as np


RNG = np.random.default_rng(20260810)
N = 5


def random_complex(shape):
    return RNG.normal(size=shape) + 1j * RNG.normal(size=shape)


def random_su5():
    q, r = np.linalg.qr(random_complex((N, N)))
    phases = np.diag(r)
    q = q @ np.diag(np.conj(phases) / np.abs(phases))
    det = np.linalg.det(q)
    return q / det ** (1 / N)


def sample():
    h = random_complex((N, 1))
    raw_delta = random_complex((N, N))
    delta = (raw_delta + raw_delta.T) / 2
    raw_sigma = random_complex((N, N))
    sigma = (raw_sigma + raw_sigma.conj().T) / 2
    sigma -= np.trace(sigma) * np.eye(N) / N
    return h, delta, sigma


def invariants(h, delta, sigma):
    dd = delta @ delta.conj().T
    norm_h = (h.conj().T @ h).item().real
    norm_delta = np.trace(dd).real
    rho_sigma_delta = sigma @ delta + delta @ sigma.T
    return np.array(
        [
            norm_h * norm_delta,
            (h.conj().T @ dd @ h).item().real,
            np.trace(delta.conj().T @ rho_sigma_delta).real,
            norm_delta**2,
            np.trace(dd @ dd).real,
        ]
    )


max_invariance_error = 0.0
rows = []
for _ in range(24):
    h, delta, sigma = sample()
    before = invariants(h, delta, sigma)
    u = random_su5()
    after = invariants(u @ h, u @ delta @ u.T, u @ sigma @ u.conj().T)
    max_invariance_error = max(max_invariance_error, float(np.max(np.abs(before - after))))
    rows.append(before)

matrix = np.vstack(rows)
normalized = matrix / np.maximum(np.linalg.norm(matrix, axis=0), 1e-15)
support_rank = int(np.linalg.matrix_rank(normalized, tol=1e-10))

checks = {
    "five_required_candidate_invariants_constructed": matrix.shape[1] == 5,
    "simultaneous_su5_invariance": max_invariance_error < 1e-9,
    "candidate_invariants_are_functionally_independent_on_samples": support_rank == 5,
    "portal_family_has_more_than_one_independent_invariant": support_rank >= 2,
    "adjoint_breaking_descriptor_is_nonzero_on_generic_samples": bool(np.any(np.abs(matrix[:, 2]) > 1e-8)),
    "scalar15_quartic_family_is_not_one_dimensional": bool(
        np.linalg.matrix_rank(normalized[:, 3:5], tol=1e-10) == 2
    ),
}

report = {
    "audit": "P4-NU15RCS-C1/N1",
    "status": "PASS" if all(checks.values()) else "FAIL",
    "occupied_carriers": ["5_H", "15_H=Sym^2(5)", "24_B"],
    "tested_invariant_support": [
        "(H^dagger H) Tr(Delta Delta^dagger)",
        "H^dagger Delta Delta^dagger H",
        "<Delta, rho_15(Sigma) Delta>",
        "[Tr(Delta Delta^dagger)]^2",
        "Tr[(Delta Delta^dagger)^2]",
    ],
    "max_su5_invariance_error": max_invariance_error,
    "sample_support_rank": support_rank,
    "checks": checks,
    "claim_boundary": (
        "Local renormalizable SU(5) recovery, counterterm closure, and absence of "
        "an additional forbidding symmetry select the invariant operator support, "
        "not the numerical coefficient vector. The result is a class-minimal "
        "working-completion selector, not a Nature-level derivation of boundary values."
    ),
}

out = Path(__file__).with_suffix(".json")
out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
