#!/usr/bin/env python3
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paperVIIC_submission_source"


def run(*args):
    print("$", " ".join(map(str, args)))
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True)


def build_archive():
    excluded = {".aux", ".bbl", ".blg", ".fdb_latexmk", ".fls", ".log", ".out", ".pdf", ".toc"}
    with zipfile.ZipFile(ROOT / "arxiv_submission_source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(SOURCE.rglob("*")):
            if path.is_file() and path.suffix not in excluded:
                archive.write(path, path.relative_to(SOURCE))


def main():
    py = sys.executable
    for script in [
        "tau_core_p4_scalar15_occupation_source_audit_v01.py",
        "tau_core_p4_scalar15_cubic_origin_audit_v01.py",
        "tau_core_p4_two_higgs_seed_edge_selection_audit_v01.py",
        "tau_core_p4_scalar15_cw_capacity_audit_v01.py",
        "tau_core_p4_scalar15_rg_identifiability_audit_v01.py",
        "tau_core_p4_scalar15_boundary_packet_origin_audit_v01.py",
        "tau_core_p4_scalar15_radiative_support_selector_audit_v01.py",
        "tau_core_p4_scalar15_rg_boundary_ray_no_go_audit_v01.py",
        "tau_core_p4_scalar15_spectral_tensor_boundary_audit_v01.py",
        "tau_core_p4_scalar15_physical_rg_completion_boundary_audit_v01.py",
        "tau_core_p4_scalar15_four_origin_assembly_audit_v01.py",
        "tau_core_p4_scalar15_physical_completion_closure_audit_v01.py",
    ]:
        run(py, f"scripts/{script}")
    run("latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
        "-cd", "paperVIIC_submission_source/main.tex")
    shutil.copy2(SOURCE / "main.pdf", ROOT / "paperVIIC.pdf")
    build_archive()
    run(py, "-m", "pytest", "-q")
    print("TECHNICAL_PAPER_VIIC_REPRODUCTION_COMPLETE")


if __name__ == "__main__":
    main()
