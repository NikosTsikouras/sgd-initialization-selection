#!/usr/bin/env python3
"""Verify table outputs against an independent final-paper numeric snapshot.

Checks the rendered TeX row values, not only a generator's own JSON. By default
regenerates tables in a temporary directory from --data-dir before checking.
Use --tables-dir to check existing generated files without regenerating them.
The fixed snapshot records the final paper: new stochastic reruns can legitimately
differ; this command tests reproduction of the published saved results.
"""
from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path

from make_tables import ROOT, generate_tables, write_tables


def numbers(text):
    return [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)]


def tex_rows(path):
    text = path.read_text()
    start = text.index(r"\midrule") + len(r"\midrule")
    end = text.index(r"\bottomrule", start)
    return [[cell.strip() for cell in line.removesuffix(r"\\").split("&")]
            for line in text[start:end].splitlines() if "&" in line]


def verify(tables_dir, snapshot):
    report = []
    for expected in snapshot["tables"]:
        path = tables_dir / (expected["filename"] + ".tex")
        rows = tex_rows(path)
        if len(rows) != len(expected["numeric_rows"]):
            raise AssertionError(f"{path.name}: expected {len(expected['numeric_rows'])} rows, found {len(rows)}")
        measured = []
        for index, (cells, wanted) in enumerate(zip(rows, expected["numeric_rows"])):
            # Table 2's condition labels contain scale factors/percentages, which
            # are labels rather than measured columns; the snapshot starts at Fits.
            selected = cells[1:] if expected["number"] == 2 else cells
            actual = numbers(" ".join(selected))
            if actual != wanted:
                raise AssertionError(f"Table {expected['number']} row {index + 1}: {actual} != {wanted}")
            measured.append(actual)
        # The machine-readable display cells must agree with the actual TeX.
        machine = json.loads(path.with_suffix(".json").read_text())
        if [r["cells"] for r in machine["rows"]] != rows:
            raise AssertionError(f"{path.name}: JSON display cells differ from TeX")
        report.append(dict(table=expected["number"], rows=len(rows),
                           numeric_entries=sum(len(values) for values in measured)))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "reference_results")
    parser.add_argument("--tables-dir", type=Path,
                        help="Check these generated TeX/JSON files instead of generating temporary ones")
    parser.add_argument("--expected", type=Path, default=ROOT / "validation/expected_tables.json")
    parser.add_argument("--report", type=Path, help="Optional JSON verification report")
    args = parser.parse_args()
    snapshot = json.loads(args.expected.read_text())
    if args.tables_dir:
        checks = verify(args.tables_dir, snapshot)
    else:
        with tempfile.TemporaryDirectory(prefix="sgd-paper-tables-") as folder:
            output = Path(folder)
            write_tables(generate_tables(args.data_dir.resolve()), output)
            checks = verify(output, snapshot)
    report = dict(result="PASS", checks=checks,
                  rows=sum(c["rows"] for c in checks),
                  numeric_entries=sum(c["numeric_entries"] for c in checks))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {len(checks)} tables, {report['rows']} rows, {report['numeric_entries']} numeric entries match the final paper.")


if __name__ == "__main__":
    main()
