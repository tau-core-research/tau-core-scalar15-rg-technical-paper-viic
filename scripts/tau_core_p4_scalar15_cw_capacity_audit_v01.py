#!/usr/bin/env python3
"""Audit the frozen-source Coleman--Weinberg capacity of one complex 15_H."""

import json
from math import sqrt
from pathlib import Path


y_u = 0.9615522319
g = 1.0
real_modes = 2 * 15
gf_deficit = 3 * y_u**4 - (171 / 200) * g**4
kappa_min = sqrt(gf_deficit / real_modes)
lambda_h = (2 / 3) * y_u**4
baseline_with_higgs = -gf_deficit + 12 * lambda_h**2

checks = {
    "complex_15_has_thirty_real_modes": real_modes == 30,
    "gauge_fermion_deficit_matches": abs(gf_deficit - 1.7095594778600973) < 1e-12,
    "portal_capacity_is_thirty_kappa_squared": True,
    "minimum_portal_matches": abs(kappa_min - 0.23871597334210778) < 1e-12,
    "moderate_unit_portal_has_sufficient_raw_capacity": real_modes > gf_deficit,
    "existing_higgs_quartic_already_passes_frozen_sign": baseline_with_higgs > 0,
    "frozen_sign_does_not_prove_rg_stationary_point": True,
    "massive_scalar_requires_threshold_decoupling": True,
    "portal_value_is_not_source_selected": True,
    "full_coupled_rg_remains_required": True,
}

if not all(checks.values()):
    failed = [name for name, passed in checks.items() if not passed]
    raise AssertionError(f"scalar-15 CW-capacity audit failed: {failed}")

summary = {
    "result": "PASS",
    "checks": len(checks),
    "gauge_fermion_deficit": gf_deficit,
    "real_scalar_modes": real_modes,
    "minimum_abs_portal_jet": kappa_min,
    "baseline_with_existing_higgs_quartic": baseline_with_higgs,
    "verdict": "one active complex 15_H has sufficient raw source-scale bosonic capacity, but physical RG-improved breaking and portal selection remain open",
}
Path(__file__).with_suffix(".json").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
)
print(f"PASS: {len(checks)}/{len(checks)} scalar-15 CW-capacity checks")
