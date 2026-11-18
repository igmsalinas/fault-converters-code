"""Tests for the format-agnostic variable-load FRA manuscript integrity audit."""

from __future__ import annotations

from pathlib import Path

from scripts.audit_variable_load_manuscript import (
    PAPER_DIR,
    RESULTS_DIR,
    audit_variable_load_manuscript,
    parse_bib_keys,
    parse_markdown_citations,
)


def test_citation_and_bib_parsers() -> None:
    bib = "@article{alpha2020,\n  title={A}\n}\n@inproceedings{beta2021,\n  title={B}\n}\n"
    keys, dups = parse_bib_keys(bib)
    assert keys == ["alpha2020", "beta2021"]
    assert dups == []

    md = "Prior work [@alpha2020] and joint studies [@alpha2020; @beta2021] show..."
    assert parse_markdown_citations(md) == {"alpha2020", "beta2021"}


def test_audit_catches_leaked_terms_and_missing_citations(tmp_path: Path) -> None:
    (tmp_path / "figures").mkdir()
    for i in range(9):
        (tmp_path / "figures" / f"f{i}.png").write_bytes(b"png")
        (tmp_path / "figures" / f"f{i}.pdf").write_bytes(b"pdf")
    (tmp_path / "tables").mkdir()
    for i in range(1, 8):
        for ext in ("csv", "md", "tex"):
            (tmp_path / "tables" / f"table0{i}_mock.{ext}").write_text("x", encoding="utf-8")
    (tmp_path / "references.bib").write_text("@article{k1,\n title={T}\n}\n", encoding="utf-8")
    figs = "\n".join(f"![fig](figures/f{i}.png)" for i in range(9))
    (tmp_path / "manuscript.md").write_text(
        f"Test [@k1; @missing_key] with adapt_load and FRA denoising TODO.\n{figs}\n",
        encoding="utf-8",
    )
    res = audit_variable_load_manuscript(paper_dir=tmp_path, results_dir=tmp_path)
    joined = " | ".join(res["errors"])
    assert "TODO" in joined
    assert "adapt_load" in joined
    assert "FRA denoising" in joined
    assert "missing_key" in joined


def test_audit_catches_missing_figures_and_standalone_tables(tmp_path: Path) -> None:
    (tmp_path / "figures").mkdir()
    for i in range(8):
        (tmp_path / "figures" / f"f{i}.png").write_bytes(b"png")
        (tmp_path / "figures" / f"f{i}.pdf").write_bytes(b"pdf")
    (tmp_path / "tables").mkdir()
    (tmp_path / "tables" / "table01_mock.csv").write_text("x", encoding="utf-8")
    (tmp_path / "references.bib").write_text("@article{k1,\n title={T}\n}\n", encoding="utf-8")
    figs = "\n".join(f"![fig](figures/f{i}.png)" for i in range(8))
    (tmp_path / "manuscript.md").write_text(f"Clean [@k1].\n{figs}\n", encoding="utf-8")
    res = audit_variable_load_manuscript(paper_dir=tmp_path, results_dir=tmp_path)
    joined = " | ".join(res["errors"])
    assert "9" in joined
    assert "table01" in joined or "table02" in joined


def test_variable_load_manuscript_passes_full_audit() -> None:
    res = audit_variable_load_manuscript(paper_dir=PAPER_DIR, results_dir=RESULTS_DIR)
    assert res["errors"] == [], f"Manuscript audit failed: {res['errors']}"
    assert res["warnings"] == [], f"Uncited bibliography warnings: {res['warnings']}"
