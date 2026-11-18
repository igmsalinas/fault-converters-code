#!/usr/bin/env python3
"""Export `papers/variable_load_fra_ad/` as a clean Overleaf-ready IEEEtran project."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.overleaf_export import export_overleaf_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "papers" / "variable_load_fra_ad",
        help="Source manuscript directory inside fault_converters.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/nacho/projects/variable-load-fra-paper"),
        help="Destination Overleaf project directory.",
    )
    args = parser.parse_args()

    summary = export_overleaf_bundle(source_dir=args.source, output_dir=args.output)
    print(f"Exported Overleaf bundle to: {summary['output_dir']}")
    print(f"Figures copied: {summary['n_figures']} | Tables mapped: {summary['n_tables']}")
    if summary["errors"]:
        print("Validation errors:")
        for err in summary["errors"]:
            print(f" - {err}")
        return 1
    print("Overleaf validation PASSED (0 errors).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
