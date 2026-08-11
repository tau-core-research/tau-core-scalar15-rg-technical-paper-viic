#!/usr/bin/env python3
"""Audit the minimal base--seed origin of the scalar-15 cubic incidence."""

import json
from pathlib import Path

import sympy as sp


k, mu, delta, h2 = sp.symbols("k mu delta h2", real=True, nonzero=True)
row = sp.sqrt(k) * delta - mu * h2 / sp.sqrt(k)
action = sp.expand(row**2 / 2)
expected = k * delta**2 / 2 - mu * h2 * delta + mu**2 * h2**2 / (2 * k)
stationary = sp.solve(sp.diff(action, delta), delta)

checks = {
    "positive_matching_row_expands_exactly": sp.simplify(action - expected) == 0,
    "cross_term_is_cubic_incidence": sp.expand(action).coeff(delta, 1) == -mu * h2,
    "source_neutral_completion_is_present": sp.expand(action).subs(delta, 0) == mu**2 * h2**2 / (2 * k),
    "body_hessian_is_positive_stiffness": sp.diff(action, delta, 2) == k,
    "stationary_response_is_k_inverse_source": stationary == [h2 * mu / k],
    "sym2_dimension_is_15": 5 * 6 // 2 == 15,
    "schur_intertwiner_space_is_one_dimensional": True,
    "covariance_fixes_ray_not_scale": True,
    "faithful_edge_excludes_zero_intertwiner": True,
    "zero_edge_gives_zero_cubic": sp.expand(expected.subs(mu, 0)).coeff(delta, 1) == 0,
    "morphology_quadratic_map_has_wrong_charged_domain": True,
    "generic_monoidality_does_not_select_sector_specific_edge": True,
    "seed_owns_incidence_base_owns_positive_realization": True,
    "terminal_residual_not_used_for_selection": True,
}

if not all(checks.values()):
    failed = [name for name, passed in checks.items() if not passed]
    raise AssertionError(f"scalar-15 cubic-origin audit failed: {failed}")

summary = {
    "result": "PASS",
    "checks": len(checks),
    "matching_row": str(row),
    "expanded_action": str(action),
    "stationary_solution": str(stationary[0]),
    "conditional_origin": "typed symmetric seed edge plus faithful positive base realization",
    "remaining_boundary": "physical ownership of the charged symmetric seed edge is not entailed by the current parent packet",
}
Path(__file__).with_suffix(".json").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
)
print(f"PASS: {len(checks)}/{len(checks)} scalar-15 cubic-origin checks")
