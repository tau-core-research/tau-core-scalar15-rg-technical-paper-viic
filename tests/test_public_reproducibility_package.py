import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_publication_files_exist():
    for path in [
        "README.md",
        "CITATION.cff",
        "paperVIIC.pdf",
        "arxiv_submission_source.zip",
        "paperVIIC_submission_source/main.tex",
        "paperVIIC_submission_source/source_and_rg_details.tex",
        "paperVIIC_submission_source/closure_and_controls.tex",
        "paperVIIC_submission_source/scalar15_main.tex",
        "paperVIIC_submission_source/refs.bib",
    ]:
        assert (ROOT / path).exists(), path


def test_scope_and_claim_boundary():
    text = "\n".join(
        (ROOT / "paperVIIC_submission_source" / name).read_text()
        for name in ["main.tex", "source_and_rg_details.tex", "closure_and_controls.tex", "scalar15_main.tex"]
    )
    for marker in [
        "Type does not imply occupation",
        "Four data classes that must not be conflated",
        "Conditional common origin of the four classes",
        "Bosonic capacity is a viability test, not selection",
        "Equal-upstream RG counterfamily",
        "Exact finite stopping point",
        "unrestricted physical parent",
        "Imported descriptor-migration and response boundary",
        "D_{OS}^{\\rm op}=Q_{OS,\\delta}(\\Xi_{OS}^{\\rm cont})",
        "\\Gamma=-H^{-1}B",
    ]:
        assert marker in text
    assert "predicts the SI value" not in text
    assert "Nature necessarily" not in text


def test_audits_are_machine_readable_and_pass():
    files = sorted((ROOT / "data" / "derived").glob("*.json"))
    assert len(files) >= 12
    for path in files:
        data = json.loads(path.read_text())
        status = data.get("status")
        if status is not None:
            assert str(status).upper() in {"PASS", "CLOSED", "CONDITIONAL_PASS"}, (path, status)


def test_key_quantitative_controls():
    capacity = json.loads((ROOT / "data/derived/tau_core_p4_scalar15_cw_capacity_audit_v01.json").read_text())
    raw = json.dumps(capacity)
    assert "0.238" in raw
    rg = json.loads((ROOT / "data/derived/tau_core_p4_scalar15_rg_identifiability_audit_v01.json").read_text())
    assert "7/6" in json.dumps(rg) or "1.166" in json.dumps(rg)


def test_arxiv_archive_is_self_contained():
    with zipfile.ZipFile(ROOT / "arxiv_submission_source.zip") as archive:
        names = set(archive.namelist())
    for name in ["main.tex", "source_and_rg_details.tex", "closure_and_controls.tex", "scalar15_main.tex", "refs.bib"]:
        assert name in names
