#!/usr/bin/env python3
"""Compare annual trends for catalogue-marked fiction and all other records.

The input is the checksum-bound annual aggregate produced by the full-corpus
trend analysis. No work-level record is read or emitted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_ROOT = ROOT / "openlibrary/data/temporal"
DEFAULT_OUTPUT_ROOT = ROOT / "results/figures"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_bound_rows(report_root: Path) -> list[dict[str, str]]:
    complete_path = report_root / "COMPLETE.json"
    annual_path = report_root / "annual_counts.csv"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if complete.get("status") != "complete":
        raise ValueError("Full-corpus trend completion marker is invalid")
    if complete.get("rows_scanned") != 13_754_720:
        raise ValueError("Full-corpus trend row count drift")
    expected = complete.get("artifacts", {}).get("annual_counts.csv")
    if expected != file_record(annual_path):
        raise ValueError("Annual trend aggregate drifted from its completion marker")
    with annual_path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def annual_series(rows: list[dict[str, str]], population: str) -> list[dict[str, str]]:
    selected = sorted(
        (
            row
            for row in rows
            if row["population"] == population and row["field"] == "any"
        ),
        key=lambda row: int(row["year"]),
    )
    if [int(row["year"]) for row in selected] != list(range(1990, 2026)):
        raise ValueError(f"Incomplete annual series: {population}")
    return selected


def render(rows: list[dict[str, str]], output_root: Path) -> dict[str, Any]:
    all_rows = annual_series(rows, "all_works")
    fiction_rows = annual_series(rows, "catalog_fiction")
    years = [int(row["year"]) for row in all_rows]
    if years != [int(row["year"]) for row in fiction_rows]:
        raise ValueError("Annual population years do not align")

    fiction_works = [int(row["works"]) for row in fiction_rows]
    fiction_matches = [int(row["explicit_count"]) for row in fiction_rows]
    all_works = [int(row["works"]) for row in all_rows]
    all_matches = [int(row["explicit_count"]) for row in all_rows]
    other_works = [total - subset for total, subset in zip(all_works, fiction_works)]
    other_matches = [
        total - subset for total, subset in zip(all_matches, fiction_matches)
    ]

    if sum(all_works) != 13_754_720 or sum(all_matches) != 16_791:
        raise ValueError("All-work totals drifted from the completed analysis")
    if sum(fiction_works) != 1_301_578 or sum(fiction_matches) != 365:
        raise ValueError("Catalogue-fiction totals drifted from the completed analysis")
    if sum(other_works) != 12_453_142 or sum(other_matches) != 16_426:
        raise ValueError("Derived other-record totals are invalid")
    for year, total_n, subset_n, total_y, subset_y in zip(
        years, all_works, fiction_works, all_matches, fiction_matches
    ):
        if not 0 <= subset_n <= total_n or not 0 <= subset_y <= total_y:
            raise ValueError(f"Invalid subset bounds in {year}")

    fiction_percent = [
        100.0 * matches / works
        for matches, works in zip(fiction_matches, fiction_works)
    ]
    other_percent = [
        100.0 * matches / works for matches, works in zip(other_matches, other_works)
    ]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10.0,
            "axes.labelsize": 10.0,
            "axes.titlesize": 10.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 9.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    output_root.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharex=True)
    for axis, values, title in (
        (axes[0], fiction_percent, "Catalogue-marked fiction"),
        (axes[1], other_percent, "Other records"),
    ):
        axis.plot(years, values, color="#1F5A7A", linewidth=2.0)
        axis.set_xlim(1990, 2025)
        axis.set_ylim(bottom=0)
        axis.set_xticks([1990, 2000, 2010, 2020])
        axis.set_xlabel("Year")
        axis.set_ylabel("Matching records (%)")
        axis.set_title(title)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.65)
        axis.set_axisbelow(True)
    figure.tight_layout()

    outputs: dict[str, Any] = {}
    for suffix, settings in (
        (
            "pdf",
            {
                "metadata": {
                    "Creator": "CROW catalogue-fiction comparison v1",
                    "CreationDate": None,
                    "ModDate": None,
                }
            },
        ),
        ("png", {"dpi": 300, "metadata": {"Software": "Matplotlib"}}),
    ):
        destination = output_root / f"catalog_fiction_vs_other_relative_v1.{suffix}"
        temporary = destination.with_name(
            destination.stem + ".tmp" + destination.suffix
        )
        figure.savefig(temporary, bbox_inches="tight", **settings)
        os.replace(temporary, destination)
        outputs[destination.name] = file_record(destination)
    plt.close(figure)
    return {
        "outputs": outputs,
        "annual_years": [years[0], years[-1]],
        "catalog_fiction": {
            "records": sum(fiction_works),
            "matches": sum(fiction_matches),
        },
        "other_records": {"records": sum(other_works), "matches": sum(other_matches)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report-root",
        type=Path,
        default=DEFAULT_REPORT_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    args = parser.parse_args()
    report_root = args.report_root.resolve()
    output_root = args.output_root.resolve()
    rows = load_bound_rows(report_root)
    result = render(rows, output_root)
    provenance = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "path": str(report_root / "annual_counts.csv"),
            **file_record(report_root / "annual_counts.csv"),
        },
        "source_complete": {
            "path": str(report_root / "COMPLETE.json"),
            **file_record(report_root / "COMPLETE.json"),
        },
        "script": {
            "path": str(Path(__file__).resolve()),
            **file_record(Path(__file__)),
        },
        **result,
    }
    marker = output_root / "catalog_fiction_vs_other_v1.json"
    atomic_json(marker, provenance)
    print(
        json.dumps(
            {"status": "complete", "marker": str(marker), **result},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
