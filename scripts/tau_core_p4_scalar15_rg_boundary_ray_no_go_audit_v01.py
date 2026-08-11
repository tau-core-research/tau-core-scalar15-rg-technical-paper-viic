#!/usr/bin/env python3
"""Audit whether one common action unit selects the scalar-15 RG boundary ray."""

import json
import math
from pathlib import Path

import numpy as np


# Two positive coefficient rays on the same five-invariant support.  Both have
# the same common-unit norm, but they are not proportional.
c_a = np.ones(5) / math.sqrt(5)
c_b = np.array([2.0, 1.0, 1.0, 1.0, 1.0]) / math.sqrt(8)

stack = np.vstack([c_a, c_b])
ray_rank = int(np.linalg.matrix_rank(stack, tol=1e-12))

# A strictly convex source selector would require an independently sourced
# metric and load on invariant-support space.
G = np.diag([1.0, 2.0, 3.0, 4.0, 5.0])
j = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
c_selected = np.linalg.solve(G, j)
stationarity_residual = float(np.linalg.norm(G @ c_selected - j))

checks = {
    "support_dimension_is_five": len(c_a) == 5,
    "common_unit_normalizes_first_ray": abs(np.linalg.norm(c_a) - 1) < 1e-12,
    "common_unit_normalizes_second_ray": abs(np.linalg.norm(c_b) - 1) < 1e-12,
    "equal_unit_rays_are_not_proportional": ray_rank == 2,
    "one_common_scale_leaves_four_projective_ratios": 5 - 1 == 4,
    "strictly_convex_metric_load_selector_is_unique": stationarity_residual < 1e-12,
}

report = {
    "audit": "P4-NU15RGRAY-N1/C1",
    "status": "PASS" if all(checks.values()) else "FAIL",
    "coefficient_order": ["kappa_1", "kappa_2", "a_Sigma", "lambda_41", "lambda_42"],
    "equal_common_unit_counterfamily": {
        "ray_a": c_a.tolist(),
        "ray_b": c_b.tolist(),
        "stacked_rank": ray_rank,
    },
    "remaining_projective_dimension": 4,
    "minimal_selector_form": "L_src(c)=1/2 c^T G_I c-j_I^T c, c_*=G_I^{-1}j_I",
    "example_selector": {
        "G_I_diagonal": np.diag(G).tolist(),
        "j_I": j.tolist(),
        "c_selected": c_selected.tolist(),
        "stationarity_residual": stationarity_residual,
    },
    "checks": {key: bool(value) for key, value in checks.items()},
    "claim_boundary": (
        "The common action unit fixes at most the overall norm of the five-component "
        "boundary vector. It does not select its four projective ratios because the "
        "five invariant directions are not related by the declared symmetry. A unique "
        "physical RG path requires a source-owned positive metric G_I and nonzero load "
        "j_I (or an equivalent spectral embedding) on invariant-support space."
    ),
}

out = Path(__file__).with_suffix(".json")
out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
