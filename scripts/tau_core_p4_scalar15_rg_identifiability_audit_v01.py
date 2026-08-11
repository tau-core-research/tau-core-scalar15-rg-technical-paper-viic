#!/usr/bin/env python3
"""Audit the source-identifiable part of the one-loop scalar-15 RG extension.

This is deliberately not labelled a full scalar-potential RGE.  The current
Tau source packet fixes the representation and hence the gauge contribution,
but it does not fix the portal, scalar self-couplings, Yukawa matrix, or the
three Standard-Model component thresholds.  The scan below therefore tests a
finite source-frozen portal/threshold envelope and proves non-uniqueness of a
putative single "full" trajectory.
"""

import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


PI = math.pi
LOOP = 16 * PI**2
Y0 = 0.9615522319
Z0 = np.array([1.0, 1.0, 1.0, Y0, 2 * Y0**4 / 3])

# One-generation baseline plus one active complex SU(5) scalar 15.  A complete
# multiplet shifts all three GUT-normalized coefficients by the same 7/6.
B_BASE = np.array([43 / 30, -35 / 6, -29 / 3])
DELTA_B_15 = np.array([7 / 6, 7 / 6, 7 / 6])
B_ACTIVE = B_BASE + DELTA_B_15


def beta(t, z, kappa, threshold_t):
    g1, g2, g3, y, lam = z
    active = t >= threshold_t
    b1, b2, b3 = B_ACTIVE if active else B_BASE
    gy2 = (3 / 5) * g1**2
    portal = 30 * kappa**2 if active else 0.0
    return np.array(
        [
            b1 * g1**3 / LOOP,
            b2 * g2**3 / LOOP,
            b3 * g3**3 / LOOP,
            y
            * (4.5 * y**2 - 8 * g3**2 - 2.25 * g2**2 - (17 / 12) * gy2)
            / LOOP,
            (
                24 * lam**2
                - 6 * y**4
                + (12 * y**2 - 9 * g2**2 - 3 * gy2) * lam
                + (9 / 8) * g2**4
                + (3 / 4) * g2**2 * gy2
                + (3 / 8) * gy2**2
                + portal
            )
            / LOOP,
        ]
    )


def perturbative_boundary(_t, z):
    return 4 * PI - max(x**2 for x in z[:4])


perturbative_boundary.terminal = True
perturbative_boundary.direction = -1


def run_case(kappa, threshold_t):
    rhs = lambda t, z: beta(t, z, kappa, threshold_t)
    solution = solve_ivp(
        rhs,
        (0, -20),
        Z0,
        events=perturbative_boundary,
        dense_output=True,
        max_step=0.01,
        rtol=2e-10,
        atol=2e-12,
    )
    end_t = float(solution.t_events[0][0]) if len(solution.t_events[0]) else -20.0
    grid = np.linspace(0, end_t, 2001)
    states = np.array([solution.sol(t) for t in grid])
    fvals = np.array([z[4] + rhs(t, z)[4] / 4 for t, z in zip(grid, states)])
    sign_changes = np.where(np.signbit(fvals[:-1]) != np.signbit(fvals[1:]))[0]
    return {
        "kappa_frozen": kappa,
        "threshold_t": threshold_t,
        "threshold_mu_over_mu0": math.exp(threshold_t),
        "end_t": end_t,
        "end_mu_over_mu0": math.exp(end_t),
        "perturbative_boundary_reached": bool(len(solution.t_events[0])),
        "minimum_F": float(np.min(fvals)),
        "maximum_F": float(np.max(fvals)),
        "stationary_zero_detected": bool(len(sign_changes)),
        "lambda_end": float(states[-1, 4]),
        "couplings_end": [float(x) for x in states[-1]],
    }


cases = [
    run_case(kappa, threshold_t)
    for threshold_t in (0.0, -2.0, -5.0)
    for kappa in (0.0, 0.24, 0.40, 0.80, 1.20, 1.50)
]

long_active_cases = [case for case in cases if case["threshold_t"] == -5.0]
same_source_different_completion = abs(
    long_active_cases[0]["lambda_end"] - long_active_cases[-1]["lambda_end"]
) > 0.1
checks = {
    "full_15_gauge_shift_is_universal": bool(np.allclose(DELTA_B_15, 7 / 6)),
    "relative_one_loop_gauge_running_unchanged_while_full_15_active": bool(
        np.allclose(
            [B_ACTIVE[0] - B_ACTIVE[1], B_ACTIVE[1] - B_ACTIVE[2]],
            [B_BASE[0] - B_BASE[1], B_BASE[1] - B_BASE[2]],
        )
    ),
    "finite_envelope_completed": len(cases) == 18,
    "all_cases_numerically_finite": all(
        math.isfinite(case["minimum_F"]) and math.isfinite(case["lambda_end"])
        for case in cases
    ),
    "same_fixed_upstream_data_admit_distinct_rg_trajectories": same_source_different_completion,
    "no_unique_full_rg_prediction_without_missing_scalar_data": same_source_different_completion,
}

report = {
    "audit": "P4-NU15RG-IDN1",
    "status": "PASS" if all(checks.values()) else "FAIL",
    "source_identified_inputs": {
        "initial_g1_g2_g3_y_lambda": Z0.tolist(),
        "baseline_gauge_coefficients": B_BASE.tolist(),
        "complete_scalar15_gauge_shift": DELTA_B_15.tolist(),
        "active_gauge_coefficients": B_ACTIVE.tolist(),
        "scalar15_real_mode_count": 30,
    },
    "unselected_full_rg_inputs": [
        "portal running and boundary value",
        "independent scalar self-couplings",
        "type-II Yukawa matrix and cubic coupling",
        "M_6, M_32, and M_3W component thresholds",
        "SU(5)-breaking threshold matching",
    ],
    "controlled_envelope": {
        "meaning": (
            "The portal is held source-frozen and a common scalar-15 threshold "
            "is scanned. This is a reduced identifiability envelope, not the "
            "unselected full scalar-potential RGE."
        ),
        "cases": cases,
    },
    "checks": checks,
    "claim_boundary": (
        "The scalar-15 representation fixes a universal one-loop gauge shift, "
        "but the current Tau source packet does not define one full RG trajectory. "
        "Different admissible portal/threshold completions with identical frozen "
        "upstream data produce distinct flows. The envelope is diagnostic only and "
        "does not select a Coleman-Weinberg vacuum or a physical scalar spectrum."
    ),
}

out = Path(__file__).with_suffix(".json")
out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
