#!/usr/bin/env python3
"""Render the manuscript aggregate annual trend and metadata-coverage figure.

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


def all_work_series(rows: list[dict[str, str]], field: str) -> list[dict[str, str]]:
    selected = sorted(
        (
            row
            for row in rows
            if row["population"] == "all_works" and row["field"] == field
        ),
        key=lambda row: int(row["year"]),
    )
    if [int(row["year"]) for row in selected] != list(range(1990, 2026)):
        raise ValueError(f"Incomplete annual all-work series: {field}")
    return selected


def render(rows: list[dict[str, str]], output_root: Path) -> dict[str, Any]:
    any_rows = all_work_series(rows, "any")
    title_rows = all_work_series(rows, "title")
    subject_rows = all_work_series(rows, "subjects")
    description_rows = all_work_series(rows, "description")
    years = [int(row["year"]) for row in any_rows]
    annual_percent = [
        100.0 * int(row["explicit_count"]) / int(row["works"]) for row in any_rows
    ]
    annual_counts = [int(row["explicit_count"]) for row in any_rows]
    peak_index = max(range(len(years)), key=annual_percent.__getitem__)
    records = sum(int(row["works"]) for row in any_rows)
    matches = sum(int(row["explicit_count"]) for row in any_rows)
    if records != 13_754_720 or matches != 16_791:
        raise ValueError("All-record totals drifted from the completed analysis")
    for index in range(len(years)):
        field_counts = [
            int(series[index]["explicit_count"])
            for series in (title_rows, subject_rows, description_rows)
        ]
        any_count = int(any_rows[index]["explicit_count"])
        if not max(field_counts) <= any_count <= sum(field_counts):
            raise ValueError(f"Invalid any-field union count in {years[index]}")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10.0,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    output_root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Any] = {}
    plot_specs = (
        ("absolute", annual_counts, "Matching records"),
        ("relative", annual_percent, "Records matching climate terms (%)"),
    )
    for name, values, y_label in plot_specs:
        figure, axis = plt.subplots(figsize=(7.2, 3.0))
        axis.plot(years, values, color="#1F5A7A", linewidth=2.0)
        axis.set_xlim(1990, 2025)
        axis.set_ylim(bottom=0)
        axis.set_xticks([1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025])
        axis.set_xlabel("Year")
        axis.set_ylabel(y_label)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.65)
        axis.set_axisbelow(True)
        figure.tight_layout()
        for suffix, settings in (
            (
                "pdf",
                {
                    "metadata": {
                        "Creator": "CROW full-corpus trends v2",
                        "CreationDate": None,
                        "ModDate": None,
                    }
                },
            ),
            ("png", {"dpi": 300, "metadata": {"Software": "Matplotlib"}}),
        ):
            destination = output_root / f"full_corpus_temporal_{name}_v2.{suffix}"
            temporary = destination.with_name(
                destination.stem + ".tmp" + destination.suffix
            )
            figure.savefig(temporary, bbox_inches="tight", **settings)
            os.replace(temporary, destination)
            outputs[destination.name] = file_record(destination)
        plt.close(figure)
    return {
        "outputs": outputs,
        "records": records,
        "matches": matches,
        "peak": {
            "year": years[peak_index],
            "percent": round(annual_percent[peak_index], 12),
        },
        "annual_years": [years[0], years[-1]],
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
    marker = output_root / "full_corpus_temporal_v2.json"
    atomic_json(marker, provenance)
    print(
        json.dumps(
            {"status": "complete", "marker": str(marker), **result},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
