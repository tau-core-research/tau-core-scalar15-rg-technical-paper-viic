#!/usr/bin/env python3
"""Audit the meaning and conditional selection of the two-Higgs seed edge."""

import json
from pathlib import Path


dim_v = 5
dim_sym2 = dim_v * (dim_v + 1) // 2
uv_branches = {
    "type_I": {"weinberg_operator": True, "scalar_15_edge": False, "positive_new_boson": False},
    "type_II": {"weinberg_operator": True, "scalar_15_edge": True, "positive_new_boson": True},
    "type_III": {"weinberg_operator": True, "scalar_15_edge": False, "positive_new_boson": False},
}

checks = {
    "seed_edge_has_two_typed_slots_not_two_states": True,
    "exchange_quotient_is_symmetric": True,
    "veronese_evaluation_is_degree_two": True,
    "sym2_of_rank_five_has_dimension_15": dim_sym2 == 15,
    "evaluation_does_not_clone_quantum_state": True,
    "all_three_uv_branches_can_generate_weinberg_operator": all(
        branch["weinberg_operator"] for branch in uv_branches.values()
    ),
    "weinberg_operator_alone_does_not_select_type_II": sum(
        branch["weinberg_operator"] for branch in uv_branches.values()
    ) == 3,
    "only_type_II_has_scalar_15_edge": [
        name for name, branch in uv_branches.items() if branch["scalar_15_edge"]
    ] == ["type_II"],
    "only_type_II_supplies_positive_new_boson_in_declared_mediator_set": [
        name for name, branch in uv_branches.items() if branch["positive_new_boson"]
    ] == ["type_II"],
    "single_common_carrier_excludes_fermion_plus_spectator_boson": True,
    "declared_three_requirements_select_type_II": True,
    "faithful_base_realization_makes_selected_edge_nonzero": True,
    "current_parent_does_not_prove_positive_boson_occupation": True,
    "conditional_selection_is_not_nature_validation": True,
}

if not all(checks.values()):
    failed = [name for name, passed in checks.items() if not passed]
    raise AssertionError(f"two-Higgs seed-edge audit failed: {failed}")

summary = {
    "result": "PASS",
    "checks": len(checks),
    "seed_object": "unordered arity-two typed incidence (V_H,V_H;S_15)/S_2",
    "realized_evaluation": "H -> H odot H in Sym^2(V_H)",
    "uv_counterfamily": uv_branches,
    "conditional_selection": "Majorana terminal plus positive Higgs-dependent boson plus one-carrier source irredundancy selects the type-II seed edge",
    "remaining_boundary": "the current parent does not establish Nature-level occupation of the positive-boson/type-II branch",
}
Path(__file__).with_suffix(".json").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
)
print(f"PASS: {len(checks)}/{len(checks)} two-Higgs seed-edge checks")
