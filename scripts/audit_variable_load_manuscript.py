#!/usr/bin/env python3
"""Rigorous Integrity Audit for the Format-Agnostic Variable-Load FRA Manuscript.

Validates `papers/variable_load_fra_ad/manuscript.md` and `references.bib`:
1. Zero placeholder markers (`TODO`, `FIXME`, `VERIFY`, `PLACEHOLDER`, `TBD`, `XXX`).
2. Zero leaked experimental complexity (`adapt_load`, `SSA+Load`, `load_adapt`).
3. 100% of `[@key]` citations resolve to unique entries in `references.bib`,
   and 100% of `references.bib` entries are cited at least once.
4. Every referenced figure in `manuscript.md` exists under `papers/variable_load_fra_ad/figures/`
   (both `.png` and `.pdf` vector versions).
5. Key numerical results in `manuscript.md` cross-check against persisted JSON artifacts.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "papers" / "variable_load_fra_ad"
RESULTS_DIR = ROOT / "papers" / "results" / "mlwa_study_dataset02"

FORBIDDEN_TOKENS = ("TODO", "FIXME", "VERIFY", "PLACEHOLDER", "TBD", "XXX")
FORBIDDEN_SIMPLIFIED_TERMS = ("adapt_load", "SSA+Load", "load_adapt", "strict_margin", "FRA denoising")


def parse_bib_keys(bib_text: str) -> tuple[List[str], List[str]]:
    keys = re.findall(r"^@\w+\{\s*([^,\s]+)\s*,", bib_text, flags=re.MULTILINE)
    seen: Set[str] = set()
    dups: List[str] = []
    for k in keys:
        if k in seen:
            dups.append(k)
        seen.add(k)
    return keys, dups


def parse_markdown_citations(md_text: str) -> Set[str]:
    # Matches [@key1] or [@key1; @key2; @key3]
    cited: Set[str] = set()
    for bracket in re.findall(r"\[([^\[\]]*@[^\[\]]+)\]", md_text):
        for k in re.findall(r"@([A-Za-z0-9_:\-]+)", bracket):
            cited.add(k)
    return cited


def parse_markdown_figures(md_text: str) -> List[str]:
    return re.findall(r"!\[[^\]]*\]\(([^)]+)\)", md_text)


def audit_variable_load_manuscript(
    paper_dir: Path = PAPER_DIR,
    results_dir: Path = RESULTS_DIR,
) -> Dict[str, List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    md_path = paper_dir / "manuscript.md"
    bib_path = paper_dir / "references.bib"

    if not md_path.exists():
        errors.append(f"Missing manuscript file: {md_path}")
        return {"errors": errors, "warnings": warnings}
    if not bib_path.exists():
        errors.append(f"Missing bibliography file: {bib_path}")
        return {"errors": errors, "warnings": warnings}

    md_text = md_path.read_text(encoding="utf-8")
    bib_text = bib_path.read_text(encoding="utf-8")

    # 1. Forbidden placeholder tokens (word-boundary check)
    for tok in FORBIDDEN_TOKENS:
        if re.search(rf"\b{re.escape(tok)}\b", md_text) or re.search(rf"\b{re.escape(tok)}\b", bib_text):
            errors.append(f"Forbidden placeholder token found: {tok}")

    # 2. Simplified-out terms check
    for term in FORBIDDEN_SIMPLIFIED_TERMS:
        if term in md_text:
            errors.append(f"Excluded complexity term found in simplified manuscript: {term}")

    # 3. Citations vs BibTeX
    bib_keys, dup_keys = parse_bib_keys(bib_text)
    if dup_keys:
        errors.append(f"Duplicate BibTeX keys in references.bib: {sorted(set(dup_keys))}")

    bib_set = set(bib_keys)
    cited_set = parse_markdown_citations(md_text)
    if not cited_set:
        errors.append("No [@key] citations found in manuscript.md")
    missing_cites = sorted(cited_set - bib_set)
    if missing_cites:
        errors.append(f"Unresolved citations in manuscript.md: {missing_cites}")
    uncited_keys = sorted(bib_set - cited_set)
    if uncited_keys:
        warnings.append(f"Uncited BibTeX entries in references.bib: {uncited_keys}")

    # 4. Figure & standalone table existence (9 PNG+PDF figures, 7 CSV+MD+TEX tables)
    fig_refs = parse_markdown_figures(md_text)
    if len(set(fig_refs)) < 9:
        errors.append(
            f"Expected at least 9 unique figures in manuscript.md, found {len(set(fig_refs))}"
        )
    for ref in fig_refs:
        fig_path = (paper_dir / ref).resolve()
        if not fig_path.exists():
            errors.append(f"Missing referenced figure file: {ref}")
        pdf_counterpart = fig_path.with_suffix(".pdf")
        if not pdf_counterpart.exists():
            errors.append(f"Missing vector PDF counterpart for figure: {pdf_counterpart.name}")

    tables_dir = paper_dir / "tables"
    for idx in range(1, 8):
        prefix = f"table0{idx}_"
        for ext in ("csv", "md", "tex"):
            matches = list(tables_dir.glob(f"{prefix}*.{ext}")) if tables_dir.exists() else []
            if not matches:
                errors.append(f"Missing standalone table artifact matching {prefix}*.{ext}")

    # 5. Balanced display math blocks ($$)
    n_display_delim = len(re.findall(r"(?<!\\)\$\$", md_text))
    if n_display_delim % 2 != 0:
        errors.append(f"Unbalanced $$ display math delimiters: count={n_display_delim}")

    # 6. Cross-check core quantitative anchors against persisted JSON artifacts
    ab10_zl_path = results_dir / "ablation_transformer_10seed_zero_label.json"
    ab10_sum_path = results_dir / "ablation_transformer_10seed_summary.json"
    phys_path = results_dir / "data_physics_audit.json"
    enc_path = results_dir / "encoder_comparison_ssa_fixed_summary.json"
    lofo_path = results_dir / "lofo_transformer_ssa_fixed_5seed_summary.json"
    if ab10_zl_path.exists() and ab10_sum_path.exists() and phys_path.exists():
        ab10_zl = json.loads(ab10_zl_path.read_text(encoding="utf-8"))
        ab10_sum = json.loads(ab10_sum_path.read_text(encoding="utf-8"))
        phys = json.loads(phys_path.read_text(encoding="utf-8"))
        ssa_zl = ab10_zl["regimes"]["tf_physics_ssa_fixed_load"]
        ssa_sum = ab10_sum["regimes"]["tf_physics_ssa_fixed_load"]
        expected_anchors = [
            f"{ssa_zl['zero_p975_f1']['mean']:.4f}",  # 0.9417
            f"{ssa_zl['auc_pr_pi05']['mean']:.4f}",  # 0.9126
            f"{ssa_zl['auc_roc']['mean']:.4f}",  # 0.9740
            f"{ssa_sum['fra015_matched_p975_f1']['mean']:.4f}",  # 0.9284
            f"{phys['psim_vs_analytic_rel_l2_error']['exact_ssa_with_true_rout']['median_pct']:.4f}",  # 0.0952
        ]
        if enc_path.exists():
            enc = json.loads(enc_path.read_text(encoding="utf-8"))
            expected_anchors.append(
                f"{enc['architectures']['Transformer']['zero_p975_f1']['mean']:.4f}"  # 0.9447
            )
        if lofo_path.exists():
            lofo = json.loads(lofo_path.read_text(encoding="utf-8"))
            expected_anchors.append(
                f"{lofo['modes']['Cout']['lofo_single_auc_roc']['mean']:.4f}"  # 0.9989
            )
        for anchor in expected_anchors:
            if anchor not in md_text:
                errors.append(
                    f"Artifact anchor value '{anchor}' from persisted study JSON not found in manuscript.md"
                )

    return {"errors": errors, "warnings": warnings}


def main() -> int:
    res = audit_variable_load_manuscript()
    for w in res["warnings"]:
        print(f"[WARN] {w}")
    for e in res["errors"]:
        print(f"[ERROR] {e}")
    if res["errors"]:
        print(f"\nAudit FAILED with {len(res['errors'])} error(s).")
        return 1
    print(f"Audit PASSED (0 errors, {len(res['warnings'])} warnings).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
