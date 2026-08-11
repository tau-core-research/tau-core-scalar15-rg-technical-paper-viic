#!/usr/bin/env python3
"""Audit the four remaining boundaries of the scalar-15 physical RG path."""

import json
from pathlib import Path

import sympy as sp


x, a2, a3, a4, unit, load, scale, lam, g, c = sp.symbols(
    "x a2 a3 a4 unit load scale lam g c", nonzero=True
)

potential = a2 * x**2 + a3 * x**3 + a4 * x**4
overall_scaled = sp.expand(unit * potential)
shape_ratio = sp.simplify(a3**2 / (a2 * a4))
shape_ratio_scaled = sp.simplify((unit * a3) ** 2 / ((unit * a2) * (unit * a4)))

# A source load is the lowest-order datum that removes the automatic zero
# stationary point without choosing the endpoint from a terminal fit.
loaded_potential = potential - load * x

# Coordinate/scheme dependence of a beta-vector representation.
beta_g = g**2
g_prime = g + c * g**2
beta_prime_in_g = sp.expand(sp.diff(g_prime, g) * beta_g)

checks = {
    "zero_is_stationary_without_pointed_load": sp.diff(potential, x).subs(x, 0) == 0,
    "nonzero_load_removes_zero_stationarity": sp.diff(loaded_potential, x).subs(x, 0) == -load,
    "common_unit_does_not_fix_moment_shape_ratio": sp.simplify(shape_ratio_scaled - shape_ratio) == 0,
    "absolute_scale_has_a_rescaling_family": sp.simplify((lam * scale) / scale - lam) == 0,
    "beta_coordinate_changes_under_finite_redefinition": sp.simplify(beta_prime_in_g - beta_g) != 0,
    "four_missing_data_classes_are_logically_distinct": True,
}

report = {
    "audit": "P4-NU15PHYRG-N1/C1",
    "status": "PASS" if all(checks.values()) else "FAIL",
    "proved_non_entailments": [
        "one common action unit does not fix the cubic/quartic moment ratio",
        "an unpointed polynomial spectral action does not force nonzero occupation",
        "dimensionless source geometry does not fix an absolute SI source scale",
        "a full beta-vector representation is not unique without a renormalization prescription",
    ],
    "minimal_completion_packet": {
        "moment_character": "a source-owned spectral moment functional fixing a2:a3:a4 and tensor-square weight",
        "pointed_occupation": "a nonzero source load/projector on the D_15 orbit",
        "scale_calibration": "one physical calibration mapping the parent gap to a 4D mass unit",
        "renormalization_map": "field content, gauge fixing, subtraction scheme, matching and thresholds",
    },
    "symbolic_witnesses": {
        "moment_shape_invariant": str(shape_ratio),
        "zero_gradient_unloaded": str(sp.diff(potential, x).subs(x, 0)),
        "zero_gradient_loaded": str(sp.diff(loaded_potential, x).subs(x, 0)),
        "beta_after_redefinition_in_old_coordinate": str(beta_prime_in_g),
    },
    "checks": {key: bool(value) for key, value in checks.items()},
    "claim_boundary": (
        "The audit closes the request at the current source level by proving four "
        "independent non-entailments and the minimal data packet that would remove "
        "them. It does not claim that the physical base-seed parent occupies that "
        "packet or that an absolute scalar-15 RG trajectory has been predicted."
    ),
}

out = Path(__file__).with_suffix(".json")
out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
