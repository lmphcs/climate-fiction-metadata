#!/usr/bin/env python3
"""Render annual corpus size and metadata-coverage diagnostics.

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
    subject_rows = all_work_series(rows, "subjects")
    description_rows = all_work_series(rows, "description")
    years = [int(row["year"]) for row in any_rows]
    works = [int(row["works"]) for row in any_rows]
    subject_coverage = [float(row["coverage_percent"]) for row in subject_rows]
    description_coverage = [float(row["coverage_percent"]) for row in description_rows]

    if sum(works) != 13_754_720:
        raise ValueError("Annual work totals drifted from the completed analysis")
    if sum(int(row["field_available"]) for row in subject_rows) != 7_196_910:
        raise ValueError("Subject/genre total drifted from the completed analysis")
    if sum(int(row["field_available"]) for row in description_rows) != 1_014_530:
        raise ValueError("Description total drifted from the completed analysis")
    for index, year in enumerate(years):
        for row in (subject_rows[index], description_rows[index]):
            if int(row["works"]) != works[index]:
                raise ValueError(f"Annual denominators disagree in {year}")
            expected = 100.0 * int(row["field_available"]) / works[index]
            if abs(float(row["coverage_percent"]) - expected) > 1e-10:
                raise ValueError(f"Coverage identity failed in {year}")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10.0,
            "axes.labelsize": 10.0,
            "axes.titlesize": 10.0,
            "legend.fontsize": 9.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True)

    axes[0].plot(
        years, [value / 1000.0 for value in works], color="#1F5A7A", linewidth=2.0
    )
    axes[0].set_title("Records by publication year")
    axes[0].set_ylabel("Records (thousands)")
    axes[0].set_ylim(bottom=0)

    axes[1].plot(
        years,
        subject_coverage,
        color="#009E73",
        linewidth=2.0,
        label="Subjects or genres",
    )
    axes[1].plot(
        years,
        description_coverage,
        color="#CC79A7",
        linewidth=2.0,
        label="Descriptions",
    )
    axes[1].set_title("Metadata availability")
    axes[1].set_ylabel("Records with field (%)")
    axes[1].set_ylim(0, 100)
    axes[1].set_xlabel("Year")
    axes[1].legend(frameon=False, loc="upper right")

    for axis in axes:
        axis.set_xlim(1990, 2025)
        axis.set_xticks([1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025])
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.65)
        axis.set_axisbelow(True)

    figure.tight_layout(h_pad=1.2)
    output_root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Any] = {}
    for suffix, settings in (
        (
            "pdf",
            {
                "metadata": {
                    "Creator": "CROW corpus coverage diagnostic v1",
                    "CreationDate": None,
                    "ModDate": None,
                }
            },
        ),
        ("png", {"dpi": 300, "metadata": {"Software": "Matplotlib"}}),
    ):
        destination = output_root / f"annual_corpus_coverage_v1.{suffix}"
        temporary = destination.with_name(
            destination.stem + ".tmp" + destination.suffix
        )
        figure.savefig(temporary, bbox_inches="tight", **settings)
        os.replace(temporary, destination)
        outputs[destination.name] = file_record(destination)
    plt.close(figure)
    return {
        "outputs": outputs,
        "records": sum(works),
        "subjects_or_genres": sum(int(row["field_available"]) for row in subject_rows),
        "descriptions": sum(int(row["field_available"]) for row in description_rows),
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
    marker = output_root / "annual_corpus_coverage_v1.json"
    atomic_json(marker, provenance)
    print(
        json.dumps(
            {"status": "complete", "marker": str(marker), **result},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
