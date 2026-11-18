#!/usr/bin/env python3
"""Reproduce the two simple vocabulary-trend plots in this package.

Only the checksum-bound compact annual aggregate is read. No work-level data is
required or emitted.
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
from matplotlib.ticker import StrMethodFormatter  # noqa: E402

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PACKAGE_ROOT / "openlibrary/data/vocabulary"
DEFAULT_OUTPUT_ROOT = PACKAGE_ROOT / "results/figures"
EXPECTED_ROWS = 13_754_720
EXPECTED_YEARS = list(range(1990, 2026))
PLOT_TERMS = {
    "climate change": "#0072B2",
    "global warming": "#D55E00",
    "climate crisis": "#009E73",
    "climatic change": "#CC79A7",
}


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


def load_verified_rows(data_root: Path) -> dict[str, list[dict[str, str]]]:
    complete_path = data_root / "COMPLETE.json"
    annual_path = data_root / "annual_term_counts.csv"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if complete.get("status") != "complete":
        raise ValueError("Completion marker status is not complete")
    if complete.get("rows_scanned") != EXPECTED_ROWS:
        raise ValueError("Unexpected full-corpus row count")
    expected = complete.get("artifacts", {}).get("annual_term_counts.csv")
    if expected != file_record(annual_path):
        raise ValueError("Annual aggregate does not match COMPLETE.json")

    with annual_path.open(encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))

    selected: dict[str, list[dict[str, str]]] = {}
    reference_works: list[int] | None = None
    for term in PLOT_TERMS:
        rows = sorted(
            (row for row in raw_rows if row["field"] == "any" and row["term"] == term),
            key=lambda row: int(row["year"]),
        )
        years = [int(row["year"]) for row in rows]
        if years != EXPECTED_YEARS:
            raise ValueError(f"Incomplete annual series for {term}")
        works = [int(row["works"]) for row in rows]
        if reference_works is None:
            reference_works = works
        elif works != reference_works:
            raise ValueError(f"Annual denominators differ for {term}")
        if any(int(row["match_count"]) > int(row["works"]) for row in rows):
            raise ValueError(f"Impossible match count for {term}")
        selected[term] = rows
    return selected


def save_figure(figure: Any, stem: Path) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    for suffix, settings in (
        (
            "pdf",
            {
                "metadata": {
                    "Creator": "CROW vocabulary diagnostic package v1",
                    "CreationDate": None,
                    "ModDate": None,
                }
            },
        ),
        ("png", {"dpi": 300, "metadata": {"Software": "Matplotlib"}}),
    ):
        destination = stem.with_suffix("." + suffix)
        temporary = destination.with_name(
            destination.stem + ".tmp" + destination.suffix
        )
        figure.savefig(temporary, bbox_inches="tight", **settings)
        os.replace(temporary, destination)
        outputs[destination.name] = file_record(destination)
    return outputs


def render(
    series: dict[str, list[dict[str, str]]], output_root: Path
) -> dict[str, Any]:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9.0,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.5,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    output_root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Any] = {}

    plot_specs = (
        (
            "absolute",
            "Climate terminology in Open Library metadata: annual counts",
            "Records containing the phrase",
            lambda row: float(row["match_count"]),
        ),
        (
            "relative",
            "Climate terminology in Open Library metadata: relative frequency",
            "Records containing the phrase (per 10,000 records)",
            lambda row: 10_000.0 * float(row["match_count"]) / float(row["works"]),
        ),
    )
    for name, title, ylabel, value in plot_specs:
        figure, axis = plt.subplots(figsize=(7.2, 3.8))
        for term, color in PLOT_TERMS.items():
            rows = series[term]
            axis.plot(
                [int(row["year"]) for row in rows],
                [value(row) for row in rows],
                color=color,
                linewidth=1.7,
                label=term,
            )
        axis.set_title(title)
        axis.set_xlabel("Publication year")
        axis.set_ylabel(ylabel)
        axis.set_xlim(1989.5, 2025.5)
        axis.set_ylim(bottom=0)
        axis.set_xticks([1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025])
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.55)
        axis.set_axisbelow(True)
        axis.legend(frameon=False, ncol=2, loc="upper left")
        if name == "absolute":
            axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        figure.tight_layout()
        outputs.update(save_figure(figure, output_root / f"vocabulary_trends_{name}"))
        plt.close(figure)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    series = load_verified_rows(data_root)
    outputs = render(series, output_root)
    provenance = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpus_rows": EXPECTED_ROWS,
        "publication_years": [EXPECTED_YEARS[0], EXPECTED_YEARS[-1]],
        "field": "any",
        "terms": list(PLOT_TERMS),
        "selection_rule": "completed terms with at least 100 matches",
        "source_annual_counts": file_record(data_root / "annual_term_counts.csv"),
        "source_complete_marker": file_record(data_root / "COMPLETE.json"),
        "script": file_record(Path(__file__).resolve()),
        "outputs": outputs,
    }
    marker = output_root / "plot_provenance.json"
    atomic_json(marker, provenance)
    print(json.dumps({"status": "complete", "outputs": outputs}, sort_keys=True))


if __name__ == "__main__":
    main()
