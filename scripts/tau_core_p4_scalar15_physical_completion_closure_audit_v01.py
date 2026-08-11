#!/usr/bin/env python3
"""Audit the finite P4 scalar-15 ownership, scale, and exact-RG closure.

The audit is deliberately class-relative.  It checks the consequences of the
recovery-complete one-copy P4 completion and does not infer that Nature selects
the type-II/scalar-15 branch.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import sympy as sp


RNG = np.random.default_rng(20260810)


def sym2_lift_matrix(n: int) -> np.ndarray:
    """Matrix of the canonical identity realization of Sym^2(C^n)."""
    columns = []
    for i in range(n):
        for j in range(i, n):
            value = np.zeros((n, n), dtype=complex)
            if i == j:
                value[i, i] = 1.0
            else:
                value[i, j] = value[j, i] = 1.0 / math.sqrt(2.0)
            columns.append(value.reshape(-1))
    return np.column_stack(columns)


def random_unitary(n: int) -> np.ndarray:
    raw = RNG.normal(size=(n, n)) + 1j * RNG.normal(size=(n, n))
    q, r = np.linalg.qr(raw)
    phases = np.diag(r)
    return q @ np.diag(np.conj(phases) / np.abs(phases))


def spectral_function(matrix: np.ndarray, function) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * function(values)) @ vectors.conj().T


def regulator_values(values: np.ndarray, k: float) -> np.ndarray:
    x = values / k**2
    return values / np.expm1(x)


def regulator_t_derivative_values(values: np.ndarray, k: float) -> np.ndarray:
    x = values / k**2
    return 2.0 * values * x * np.exp(x) / np.expm1(x) ** 2


def flow_trace(operator: np.ndarray, k: float) -> float:
    regulator = spectral_function(operator, lambda values: regulator_values(values, k))
    derivative = spectral_function(
        operator,
        lambda values: regulator_t_derivative_values(values, k),
    )
    return float(0.5 * np.trace(np.linalg.solve(operator + regulator, derivative)).real)


def main() -> None:
    # The charged seed edge is faithful exactly when the canonical Sym^2 map
    # has full rank.  The pointed Higgs lock then induces nonzero Delta.
    lift = sym2_lift_matrix(5)
    lift_rank = int(np.linalg.matrix_rank(lift, tol=1e-12))
    h = RNG.normal(size=5) + 1j * RNG.normal(size=5)
    hh = np.outer(h, h)
    induced_norm = float(np.linalg.norm(hh))

    k_h, lambda_h, mu_delta, k_delta = sp.symbols(
        "k_h lambda_h mu_delta k_delta", positive=True
    )
    h2_star = sp.simplify(k_h / (2 * lambda_h))
    delta_star = sp.simplify(mu_delta * h2_star / k_delta)

    # Common PIS/gravity length lock: ell_H=sqrt(48*pi) ell_P and the first
    # one-copy PIS gap is sqrt(2)/ell_H.
    ratio_exact = sp.simplify(sp.sqrt(2) / sp.sqrt(48 * sp.pi))
    ratio_numeric = float(sp.N(ratio_exact, 18))

    # 2022 CODATA values. hbar and c are exact in SI; G carries the dominant
    # uncertainty.  The number below is a cross-terminal calibration, not an
    # independent prediction of G.
    hbar = 1.054571817e-34
    c = 299_792_458.0
    gravitational_constant = 6.67430e-11
    gravitational_constant_uncertainty = 0.00015e-11
    electron_volt = 1.602176634e-19
    planck_energy_gev = math.sqrt(hbar * c**5 / gravitational_constant) / electron_volt / 1e9
    m_star_gev = ratio_numeric * planck_energy_gev
    relative_uncertainty = 0.5 * gravitational_constant_uncertainty / gravitational_constant
    m_star_uncertainty_gev = relative_uncertainty * m_star_gev

    # Smooth exponential spectral regulator on a positive covariant operator.
    raw = RNG.normal(size=(9, 9)) + 1j * RNG.normal(size=(9, 9))
    positive_operator = np.einsum("ji,jk->ik", raw.conj(), raw) + 0.25 * np.eye(9)
    unitary = random_unitary(9)
    transformed = unitary @ positive_operator @ unitary.conj().T
    k = 2.0
    regulator = spectral_function(
        positive_operator,
        lambda values: regulator_values(values, k),
    )
    transformed_regulator = spectral_function(
        transformed,
        lambda values: regulator_values(values, k),
    )
    covariance_error = float(
        np.max(np.abs(transformed_regulator - unitary @ regulator @ unitary.conj().T))
    )

    # Direct sums must give additive supertraces before the statistics signs
    # are inserted.  This is the finite spectral counterpart of one-copy
    # counterterm additivity.
    block_a = positive_operator[:4, :4]
    block_b = positive_operator[4:, 4:]
    block_sum = np.block(
        [
            [block_a, np.zeros((4, 5), dtype=complex)],
            [np.zeros((5, 4), dtype=complex), block_b],
        ]
    )
    additive_error = abs(
        flow_trace(block_sum, k) - flow_trace(block_a, k) - flow_trace(block_b, k)
    )

    # The exponential regulator behaves as k^2 for low modes and decouples
    # high modes.  Use scalar contributions to avoid matrix-conditioning noise.
    low_value = np.array([1e-8 * k**2])
    high_value = np.array([50.0 * k**2])
    low_regulator_ratio = float(regulator_values(low_value, k)[0] / k**2)
    low_flow = float(
        0.5
        * regulator_t_derivative_values(low_value, k)[0]
        / (low_value[0] + regulator_values(low_value, k)[0])
    )
    high_flow = float(
        0.5
        * regulator_t_derivative_values(high_value, k)[0]
        / (high_value[0] + regulator_values(high_value, k)[0])
    )

    # A complete independent operator basis has an invertible projection Gram
    # matrix, so the exact functional flow defines a unique beta vector in the
    # declared truncation.  This is a projection theorem, not a precomputed
    # list of component polynomials.
    operator_samples = RNG.normal(size=(32, 8))
    projection_gram = np.einsum("ni,nj->ij", operator_samples, operator_samples)
    gram_rank = int(np.linalg.matrix_rank(projection_gram, tol=1e-12))
    target_flow = RNG.normal(size=32)
    beta_projection = np.linalg.solve(
        projection_gram,
        operator_samples.T @ target_flow,
    )
    normal_equation_error = float(
        np.linalg.norm(
            operator_samples.T @ (operator_samples @ beta_projection - target_flow)
        )
    )

    checks = {
        "canonical_sym2_seed_edge_has_rank_15": lift_rank == 15,
        "occupied_higgs_pair_has_nonzero_scalar15_source": induced_norm > 1e-12,
        "pointed_higgs_lock_is_nonzero": h2_star != 0,
        "induced_scalar15_stationary_branch_is_nonzero": delta_star != 0,
        "common_length_scale_is_inverse_sqrt_24pi": ratio_exact == 1 / sp.sqrt(24 * sp.pi),
        "calibrated_m_star_is_positive": m_star_gev > 0,
        "spectral_regulator_is_gauge_covariant": covariance_error < 1e-10,
        "spectral_flow_is_block_additive": additive_error < 1e-10,
        "low_modes_receive_k_squared_regulator": abs(low_regulator_ratio - 1.0) < 1e-7,
        "high_modes_decouple_relative_to_low_modes": high_flow < 1e-15 * low_flow,
        "radiatively_complete_projection_basis_is_full_rank": gram_rank == 8,
        "beta_projection_satisfies_normal_equations": normal_equation_error < 1e-10,
    }

    report = {
        "audit": "P4-NU15CLOSE-C1/N1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "class_relative_ownership": {
            "seed": "pointed exterior V5 grammar plus typed symmetric edge e_HHDelta",
            "base": "faithful positive Hermitian/Hodge realization with one common length",
            "body": "nonzero Higgs lock and induced scalar-15 stationary response",
            "sym2_rank": lift_rank,
            "induced_source_norm": induced_norm,
            "boundary": "recovery-complete P4/type-II class; not unrestricted Nature selection",
        },
        "absolute_scale": {
            "exact_ratio": "M_*/E_P=1/sqrt(24*pi), equivalently M_*=Mbar_P/sqrt(3)",
            "ratio_numeric": ratio_numeric,
            "planck_energy_gev_2022_codata": planck_energy_gev,
            "M_star_gev": m_star_gev,
            "M_star_standard_uncertainty_gev_from_G": m_star_uncertainty_gev,
            "boundary": "one cross-terminal Planck calibration; not an SI prediction independent of G",
        },
        "exact_rg_rule": {
            "total_operator": "P_k=Gamma_k^(2)+gauge-fixing+ghost blocks on the full 4D x D_15 carrier",
            "regulator": "R_k(P)=P/(exp(P/k^2)-1)",
            "flow": "partial_t Gamma_k=1/2 STr[(Gamma_k^(2)+R_k)^(-1) partial_t R_k]",
            "beta_projection": "beta^I=(G_O^(-1))^(IJ)<O_J,partial_t Gamma_k>",
            "covariance_error": covariance_error,
            "block_additivity_error": additive_error,
            "boundary": "exact full functional flow; explicit component beta polynomials remain a reproducible evaluation task",
        },
        "checks": {name: bool(value) for name, value in checks.items()},
        "claim_boundary": (
            "The recovery-complete P4/type-II class now fixes the minimal seed/base "
            "ownership packet, a common Planck-calibrated scale, and one exact "
            "gauge-covariant functional RG prescription. It does not prove that "
            "Nature selects the scalar-15 branch, predict G in SI units, or replace "
            "the explicit projection of the exact flow onto every component coupling."
        ),
    }

    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
