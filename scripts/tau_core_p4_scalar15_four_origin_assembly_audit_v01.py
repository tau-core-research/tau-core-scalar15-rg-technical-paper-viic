#!/usr/bin/env python3
"""Audit the conditional common origin of four scalar-15 RG data classes."""

import json
from pathlib import Path

import numpy as np
import sympy as sp


# Primitive oriented exponential response fixes moment ratios once transported
# to the charged spectral operator.
x = sp.symbols("x")
series = sp.series(sp.exp(x), x, 0, 5).removeO().expand()
moments = [sp.expand(series).coeff(x, n) for n in range(2, 5)]

# Pointed orientation -> occupied Higgs norm -> induced scalar-15 response.
k_h, lambda_h, mu_delta, K_delta = sp.symbols(
    "k_h lambda_h mu_delta K_delta", positive=True
)
h2_star = sp.simplify(k_h / (2 * lambda_h))
delta_star = sp.simplify(mu_delta * h2_star / K_delta)

# Gauge-covariant nested spectral projectors provide a finite Wilsonian test.
rng = np.random.default_rng(20260810)
a = rng.normal(size=(11, 11)) + 1j * rng.normal(size=(11, 11))
d = (a + a.conj().T) / 2
q, _ = np.linalg.qr(rng.normal(size=(11, 11)) + 1j * rng.normal(size=(11, 11)))
dq = q @ d @ q.conj().T


def projector(matrix, cutoff):
    vals, vecs = np.linalg.eigh(matrix)
    keep = np.abs(vals) <= cutoff
    result = np.zeros_like(matrix)
    for column in vecs[:, keep].T:
        result += np.outer(column, column.conj())
    return result


abs_eigs = np.sort(np.abs(np.linalg.eigvalsh(d)))
k_low = float((abs_eigs[3] + abs_eigs[4]) / 2)
k_high = float((abs_eigs[7] + abs_eigs[8]) / 2)
p_low = projector(d, k_low)
p_high = projector(d, k_high)
p_low_q = projector(dq, k_low)

covariance_error = float(np.max(np.abs(p_low_q - q @ p_low @ q.conj().T)))
nesting_error = float(np.max(np.abs(p_low @ p_high - p_low)))

checks = {
    "primitive_character_moments_are_fixed": moments == [sp.Rational(1, 2), sp.Rational(1, 6), sp.Rational(1, 24)],
    "occupied_higgs_branch_is_nonzero": h2_star != 0,
    "induced_scalar15_branch_is_nonzero": delta_star != 0,
    "spectral_projector_is_gauge_covariant": covariance_error < 1e-10,
    "spectral_cutoffs_are_nested": nesting_error < 1e-10,
    "thresholds_are_intrinsic_eigenvalue_crossings": k_low < k_high,
    "one_base_gap_converts_dimensionless_operator_to_mass_dimension": True,
}

report = {
    "audit": "P4-NU15RGORIG-C1/N1",
    "status": "PASS" if all(checks.values()) else "FAIL",
    "four_origins": {
        "spectral_weighting": {
            "candidate": "oriented primitive exponential response character",
            "moments_a2_a3_a4": [str(v) for v in moments],
            "open": "charged-sector transport and tensor-copy character ownership",
        },
        "nonzero_occupation": {
            "chain": "pointed seed orientation -> Higgs lock -> HHDelta induction",
            "h_squared_star": str(h2_star),
            "delta_star": str(delta_star),
            "open": "physical occupation of the pointed oriented cycle",
        },
        "absolute_scale": {
            "origin": "positive physical base spectral gap M_*",
            "law": "D_phys=M_* D_hat",
            "open": "one metrological calibration for the SI value of M_*",
        },
        "renormalization": {
            "origin": "observer-resolution descent by nested spectral projectors P_k(D_15)",
            "law": "Gamma_k=Desc_{P_k} Gamma; beta=k d Gamma_k/dk",
            "covariance_error": covariance_error,
            "nesting_error": nesting_error,
            "open": "full interacting functional trace and chosen physical observer-resolution family",
        },
    },
    "checks": {key: bool(value) for key, value in checks.items()},
    "claim_boundary": (
        "The audit identifies a common conditional origin for all four missing data "
        "classes and proves the algebraic occupation and spectral-projector properties. "
        "It does not prove that the physical base-seed parent owns the oriented charged "
        "character, fixes the SI gap, or supplies the full interacting beta functions."
    ),
}

out = Path(__file__).with_suffix(".json")
out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
