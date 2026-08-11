#!/usr/bin/env python3
"""Audit what the HHDelta matching row does and does not source."""

import json
from pathlib import Path

import sympy as sp


K0, mu, h, delta, kappa, a_sigma, sigma, t_r = sp.symbols(
    "K0 mu h delta kappa a_sigma sigma t_r", real=True
)

row_action = sp.expand(sp.Rational(1, 2) * (sp.sqrt(K0) * delta - mu * h**2 / sp.sqrt(K0)) ** 2)
expected = K0 * delta**2 / 2 - mu * h**2 * delta + mu**2 * h**4 / (2 * K0)

K_enriched = K0 + kappa * h**2 + a_sigma * sigma * t_r
enriched_quadratic = sp.expand(K_enriched * delta**2 / 2)

checks = {
    "one_row_expansion_exact": sp.simplify(row_action - expected) == 0,
    "one_row_contains_scalar_stiffness": row_action.coeff(delta, 2) == K0 / 2,
    "one_row_contains_cubic": row_action.coeff(delta, 1).coeff(h, 2) == -mu,
    "one_row_contains_required_higgs_quartic_completion": row_action.coeff(h, 4) == mu**2 / (2 * K0),
    "one_row_contains_no_higgs_scalar15_portal": row_action.coeff(delta, 2).coeff(h, 2) == 0,
    "enriched_stiffness_generates_portal": enriched_quadratic.coeff(delta, 2).coeff(h, 2) == kappa / 2,
    "adjoint_background_generates_component_split": enriched_quadratic.coeff(delta, 2).coeff(sigma, 1).coeff(t_r, 1) == a_sigma / 2,
    "portal_and_split_coefficients_are_independent_of_cubic_mu": (
        sp.diff(enriched_quadratic.coeff(delta, 2).coeff(h, 2), mu) == 0
        and sp.diff(enriched_quadratic.coeff(delta, 2).coeff(sigma, 1).coeff(t_r, 1), mu) == 0
    ),
}

report = {
    "audit": "P4-NU15PACK-N1/C1",
    "status": "PASS" if all(checks.values()) else "FAIL",
    "matching_row_action": str(row_action),
    "derived_from_single_HHDelta_edge": [
        "scalar-15 quadratic stiffness",
        "HHDelta cubic incidence",
        "source-neutral Higgs quartic completion mu^2 h^4/(2 K0)",
    ],
    "not_derived_from_single_HHDelta_edge": [
        "Higgs--scalar15 portal kappa h^2 delta^2/2",
        "SU(5)-breaking component splitting a_sigma sigma t_r delta^2/2",
        "independent scalar-15 quartic invariants",
        "type-II Yukawa boundary data",
    ],
    "minimal_typed_enrichment": [
        "norm edge (V_H^* tensor V_H) tensor (15_H^* tensor 15_H) -> 1",
        "breaking edge 15_H^* tensor 24_B tensor 15_H -> 1",
        "independent invariant edges spanning Quartic(15_H)",
    ],
    "checks": {key: bool(value) for key, value in checks.items()},
    "claim_boundary": (
        "The positive HHDelta matching row fixes its own cubic and neutral quartic "
        "completion, but cannot select the portal, component thresholds, or scalar "
        "self-couplings. Those require separately typed seed/base incidences. This "
        "is a source-rank no-go, not a rejection of an enriched scalar-15 completion."
    ),
}

out = Path(__file__).with_suffix(".json")
out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
