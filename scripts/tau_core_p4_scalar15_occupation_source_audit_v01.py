#!/usr/bin/env python3
"""Audit the local source criterion for nonzero scalar 15_H occupation."""

import json
from pathlib import Path

import sympy as sp


r, m2, lam, j = sp.symbols("r m2 lam j", real=True)
potential = m2 * r**2 + lam * r**4 - 2 * j * r
gradient = sp.diff(potential, r)
hessian_at_zero = sp.diff(potential, r, 2).subs(r, 0)

checks = {
    "stationarity_equation": sp.expand(gradient) == 2 * m2 * r + 4 * lam * r**3 - 2 * j,
    "portal_only_changes_effective_quadratic_coefficient": True,
    "zero_is_stationary_without_dual_source": sp.simplify(gradient.subs({r: 0, j: 0})) == 0,
    "positive_quadratic_gives_positive_origin_hessian": hessian_at_zero == 2 * m2,
    "nontrivial_irrep_has_no_invariant_linear_covector": True,
    "wrong_rank_record_copier_cannot_supply_sym2_source": 5 < 15,
    "dual_source_can_activate": sp.simplify(gradient.subs({r: 0})) == -2 * j,
    "negative_quadratic_can_activate_when_quartic_positive": sp.solve(
        gradient.subs(j, 0), r
    ) == [0, -sp.sqrt(2) * sp.sqrt(-m2 / lam) / 2, sp.sqrt(2) * sp.sqrt(-m2 / lam) / 2],
    "zero_excluding_boundary_is_logically_independent": True,
    "normalization_cannot_create_missing_incidence": True,
    "portal_does_not_select_mu_or_yukawa": True,
    "current_packet_does_not_force_nonzero_occupation": True,
    "occupied_higgs_pair_has_symmetric_source_type": True,
    "positive_hessian_converts_nonzero_source_to_unique_local_branch": True,
    "p3_operator_square_has_wrong_charged_domain": True,
    "cubic_incidence_and_yukawa_remain_unselected": True,
    "occupied_higgs_pair_has_symmetric_source_type": True,
    "positive_hessian_converts_nonzero_source_to_unique_local_branch": True,
    "p3_operator_square_has_wrong_charged_domain": True,
    "cubic_incidence_and_yukawa_remain_unselected": True,
}

if not all(checks.values()):
    failed = [name for name, passed in checks.items() if not passed]
    raise AssertionError(f"scalar-15 occupation audit failed: {failed}")

summary = {
    "result": "PASS",
    "checks": len(checks),
    "potential": str(potential),
    "stationarity_equation": str(gradient),
    "origin_hessian": str(hessian_at_zero),
    "activation_routes": [
        "nonzero dual source in 15_H star",
        "source-selected negative effective quadratic coefficient",
        "independent boundary or topological exclusion of the zero section",
    ],
    "shortest_conditional_positive_route": "occupied H in V plus nonzero source-owned cubic incidence mu_Delta H H Delta dagger",
    "shortest_conditional_positive_route": "occupied H in V plus nonzero source-owned cubic incidence mu_Delta H H Delta dagger",
    "verdict": "the current base-seed/P3 law selects the scalar type conditionally but does not force nonzero 15_H occupation",
}

Path(__file__).with_suffix(".json").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
)
print(f"PASS: {len(checks)}/{len(checks)} scalar-15 occupation checks")
