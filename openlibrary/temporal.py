#!/usr/bin/env python3
"""Analyze explicit climate framing over time in the frozen eligible-work corpus.

The confirmatory observable is deliberately narrow: exact explicit-climate
anchors in supplied Open Library title, subject/genre, or description paratext.
The script emits aggregate counts only and never materializes candidate records.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

from openlibrary.common import (
    ROOT,
    atomic_csv,
    atomic_json,
    file_record,
    resolve_path,
    utc_now,
)

from openlibrary.statistics import rate, period_rows, grouped_binomial_trend, trend_rows

FIELDS = ("title", "subjects", "description", "any")
COLUMNS = [
    "year",
    "title",
    "subtitle",
    "subjects",
    "best_description",
    "fiction_likely",
    "year_conflict",
    "year_edition_only",
]


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def explicit_pattern(terms: Sequence[str]) -> str:
    if not terms or any(not str(term).strip() for term in terms):
        raise ValueError("Explicit-climate terms must be nonempty strings")
    alternatives = "|".join(re.escape(str(term).casefold()) for term in terms)
    return rf"(?i)\b(?:{alternatives})\b"


def arrow_strings(array: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    if pa.types.is_null(array.type):
        return pa.array([""] * len(array), type=pa.string())
    return pc.fill_null(array, "")


def boolean_numpy(array: pa.Array | pa.ChunkedArray) -> np.ndarray:
    return np.asarray(
        pc.fill_null(array, False).to_numpy(zero_copy_only=False), dtype=bool
    )


def string_available(array: pa.Array | pa.ChunkedArray) -> np.ndarray:
    return np.asarray(
        pc.greater(pc.utf8_length(arrow_strings(array)), 0).to_numpy(
            zero_copy_only=False
        ),
        dtype=bool,
    )


def field_masks(table: pa.Table, pattern: str) -> dict[str, dict[str, np.ndarray]]:
    titles = pc.binary_join_element_wise(
        arrow_strings(table["title"]),
        arrow_strings(table["subtitle"]),
        " ",
        null_handling="replace",
    )
    subjects = pc.fill_null(pc.binary_join(table["subjects"], " | "), "")
    descriptions = arrow_strings(table["best_description"])
    values = {
        "title": titles,
        "subjects": subjects,
        "description": descriptions,
    }
    output: dict[str, dict[str, np.ndarray]] = {}
    for field, strings in values.items():
        output[field] = {
            "available": string_available(strings),
            "explicit": boolean_numpy(pc.match_substring_regex(strings, pattern)),
        }
    output["any"] = {
        "available": np.logical_or.reduce(
            [output[field]["available"] for field in values]
        ),
        "explicit": np.logical_or.reduce(
            [output[field]["explicit"] for field in values]
        ),
    }
    return output


def empty_counts(
    populations: Sequence[str], first_year: int, last_year: int
) -> dict[str, dict[str, Any]]:
    size = last_year - first_year + 1
    return {
        population: {
            "works": np.zeros(size, dtype=np.int64),
            "fields": {
                field: {
                    "available": np.zeros(size, dtype=np.int64),
                    "explicit": np.zeros(size, dtype=np.int64),
                }
                for field in FIELDS
            },
        }
        for population in populations
    }


def population_masks(table: pa.Table) -> dict[str, np.ndarray]:
    rows = len(table)
    fiction = boolean_numpy(table["fiction_likely"])
    conflict = boolean_numpy(table["year_conflict"])
    edition_only = boolean_numpy(table["year_edition_only"])
    return {
        "all_works": np.ones(rows, dtype=bool),
        "catalog_fiction": fiction,
        "no_year_conflict": ~conflict,
        "work_year_supported": ~edition_only,
    }


def accumulate_batch(
    table: pa.Table,
    *,
    pattern: str,
    first_year: int,
    last_year: int,
    counts: dict[str, dict[str, Any]],
) -> int:
    years = np.asarray(table["year"].to_numpy(zero_copy_only=False), dtype=np.int64)
    in_scope = (years >= first_year) & (years <= last_year)
    if not in_scope.all():
        raise ValueError("Eligible-work shard contains a year outside frozen scope")
    indices = years - first_year
    fields = field_masks(table, pattern)
    populations = population_masks(table)
    bins = last_year - first_year + 1
    for population, pop_mask in populations.items():
        if population not in counts:
            raise ValueError(f"Unconfigured population encountered: {population}")
        counts[population]["works"] += np.bincount(
            indices, weights=pop_mask.astype(np.int64), minlength=bins
        ).astype(np.int64)
        for field in FIELDS:
            counts[population]["fields"][field]["available"] += np.bincount(
                indices,
                weights=(pop_mask & fields[field]["available"]).astype(np.int64),
                minlength=bins,
            ).astype(np.int64)
            counts[population]["fields"][field]["explicit"] += np.bincount(
                indices,
                weights=(pop_mask & fields[field]["explicit"]).astype(np.int64),
                minlength=bins,
            ).astype(np.int64)
    return len(table)


def annual_rows(
    counts: Mapping[str, Mapping[str, Any]], first_year: int, last_year: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for population, record in counts.items():
        for offset, year in enumerate(range(first_year, last_year + 1)):
            works = int(record["works"][offset])
            for field in FIELDS:
                available = int(record["fields"][field]["available"][offset])
                explicit = int(record["fields"][field]["explicit"][offset])
                rows.append(
                    {
                        "year": year,
                        "population": population,
                        "field": field,
                        "works": works,
                        "field_available": available,
                        "explicit_count": explicit,
                        "rate_per_10000_works": rate(explicit, works),
                        "rate_per_10000_available": rate(explicit, available),
                        "coverage_percent": (
                            None if works == 0 else 100.0 * available / works
                        ),
                    }
                )
    return rows


def validate_contract(
    config_path: Path, config: Mapping[str, Any]
) -> tuple[list[Path], list[str]]:
    if config.get("status") != "frozen_before_full_corpus_analysis":
        raise ValueError("Trend config is not frozen")
    input_config = config["input"]
    root = Path(input_config["eligible_root"])
    marker_path = Path(input_config["complete_marker"])
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if int(marker.get("eligible_works", -1)) != int(input_config["expected_rows"]):
        raise ValueError("Eligible-work marker row count differs from frozen contract")
    if int(marker.get("buckets", -1)) != int(input_config["expected_shards"]):
        raise ValueError(
            "Eligible-work marker shard count differs from frozen contract"
        )
    shards = sorted(root.glob("bucket-*.parquet"))
    if len(shards) != int(input_config["expected_shards"]):
        raise ValueError("Eligible-work shard count differs from frozen contract")
    terms_contract = config["lexical_contract"]
    terms_path = resolve_path(terms_contract["source_terms_file"])
    source_terms = load_yaml(terms_path)[terms_contract["source_group"]]
    frozen_terms = [str(term) for term in terms_contract["explicit_terms"]]
    if list(source_terms) != frozen_terms:
        raise ValueError("Explicit-climate terms drifted from the frozen source group")
    populations = set(config["scope"]["populations"])
    if populations != {
        "all_works",
        "catalog_fiction",
        "no_year_conflict",
        "work_year_supported",
    }:
        raise ValueError("Frozen trend populations changed")
    if set(config["scope"]["fields"]) != set(FIELDS):
        raise ValueError("Frozen trend fields changed")
    return shards, frozen_terms


def scan_shards(
    shards: Sequence[Path],
    *,
    pattern: str,
    first_year: int,
    last_year: int,
    populations: Sequence[str],
    batch_size: int,
) -> tuple[dict[str, dict[str, Any]], int]:
    counts = empty_counts(populations, first_year, last_year)
    rows = 0
    for shard in shards:
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=COLUMNS):
            rows += accumulate_batch(
                pa.Table.from_batches([batch]),
                pattern=pattern,
                first_year=first_year,
                last_year=last_year,
                counts=counts,
            )
    return counts, rows


def markdown_report(
    config: Mapping[str, Any],
    periods: Sequence[Mapping[str, Any]],
    trends: Sequence[Mapping[str, Any]],
) -> str:
    primary = next(row for row in trends if row["role"] == "primary")
    if primary["status"] != "estimated":
        finding = "The predeclared temporal slope was not estimable."
    elif primary["wald_p_two_sided"] < 0.05 and primary["odds_ratio_per_decade"] > 1:
        finding = "The predeclared test found a positive temporal association."
    elif primary["wald_p_two_sided"] < 0.05 and primary["odds_ratio_per_decade"] < 1:
        finding = "The predeclared test found a negative temporal association."
    else:
        finding = "The predeclared test found insufficient evidence of a temporal association."
    primary_periods = [
        row
        for row in periods
        if row["population"] == "all_works" and row["field"] == "any"
    ]
    lines = [
        "# Full-corpus explicit-climate framing trends v1",
        "",
        finding,
        "",
        "| Period | Works | Explicit | Rate per 10,000 works |",
        "|---|---:|---:|---:|",
    ]
    for row in primary_periods:
        value = row["rate_per_10000_works"]
        formatted = "n/a" if value is None else f"{value:.3f}"
        lines.append(
            f"| {row['period']} | {row['works']:,} | {row['explicit_count']:,} | "
            f"{formatted} |"
        )
    lines.extend(
        [
            "",
            "## Predeclared trend estimates",
            "",
            "| Test | Role | OR per decade | 95% CI | p |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in trends:
        if row["status"] == "estimated":
            lines.append(
                f"| {row['id']} | {row['role']} | {row['odds_ratio_per_decade']:.4f} | "
                f"{row['wald_95_percent_low']:.4f}-{row['wald_95_percent_high']:.4f} | "
                f"{row['wald_p_two_sided']:.4g} |"
            )
        else:
            lines.append(f"| {row['id']} | {row['role']} | n/e | n/e | n/e |")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            str(config["claim_guardrail"]),
            "",
            "The catalog-fiction subset is an automatic metadata sensitivity, not a "
            "replacement for manual fiction/nonfiction annotation. Field-available "
            "rates and coverage are reported because paratext availability changes "
            "over time. These associations are descriptive, not causal.",
            "",
        ]
    )
    return "\n".join(lines)


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def run(
    config_path: Path, output_override: Path | None, batch_size: int
) -> dict[str, Any]:
    config = load_yaml(config_path)
    shards, terms = validate_contract(config_path, config)
    scope = config["scope"]
    first_year = int(scope["first_year"])
    last_year = int(scope["last_year"])
    output = output_override or resolve_path(config["output"]["directory"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    counts, rows = scan_shards(
        shards,
        pattern=explicit_pattern(terms),
        first_year=first_year,
        last_year=last_year,
        populations=list(scope["populations"]),
        batch_size=batch_size,
    )
    if rows != int(config["input"]["expected_rows"]):
        raise ValueError(
            f"Scanned {rows} rows, expected {config['input']['expected_rows']}"
        )
    annual = annual_rows(counts, first_year, last_year)
    periods = period_rows(annual, scope["periods"])
    trends = trend_rows(annual, config["trend_tests"])
    annual_fields = [
        "year",
        "population",
        "field",
        "works",
        "field_available",
        "explicit_count",
        "rate_per_10000_works",
        "rate_per_10000_available",
        "coverage_percent",
    ]
    period_fields = [
        "period",
        "first_year",
        "last_year",
        *annual_fields[1:],
    ]
    trend_fields = list(config["trend_tests"][0].keys()) + [
        "status",
        "years",
        "successes",
        "trials",
        "odds_ratio_per_decade",
        "wald_95_percent_low",
        "wald_95_percent_high",
        "wald_z",
        "wald_p_two_sided",
        "fitted_rate_per_10000_first_year",
        "fitted_rate_per_10000_last_year",
        "model",
    ]
    atomic_csv(output / "annual_counts.csv", annual, annual_fields)
    atomic_csv(output / "period_counts.csv", periods, period_fields)
    atomic_csv(output / "trend_estimates.csv", trends, trend_fields)
    result = {
        "status": "complete",
        "created_utc": utc_now(),
        "source_snapshot": config["source_snapshot"],
        "rows_scanned": rows,
        "terms": terms,
        "period_counts": periods,
        "trend_estimates": trends,
        "claim_guardrail": config["claim_guardrail"],
    }
    atomic_json(output / "report.json", result)
    atomic_text(output / "README.md", markdown_report(config, periods, trends))
    provenance = {
        "created_utc": utc_now(),
        "config": {"path": str(config_path.resolve()), **file_record(config_path)},
        "script": {
            "path": str(Path(__file__).resolve()),
            **file_record(Path(__file__)),
        },
        "terms_source": {
            "path": str(resolve_path(config["lexical_contract"]["source_terms_file"])),
            **file_record(
                resolve_path(config["lexical_contract"]["source_terms_file"])
            ),
        },
        "input_marker": {
            "path": str(Path(config["input"]["complete_marker"])),
            **file_record(Path(config["input"]["complete_marker"])),
        },
        "input_shards": [{"path": str(path), **file_record(path)} for path in shards],
    }
    atomic_json(output / "provenance.json", provenance)
    artifact_names = [
        "annual_counts.csv",
        "period_counts.csv",
        "trend_estimates.csv",
        "report.json",
        "README.md",
        "provenance.json",
    ]
    complete = {
        "status": "complete",
        "created_utc": utc_now(),
        "rows_scanned": rows,
        "source_snapshot": config["source_snapshot"],
        "artifacts": {name: file_record(output / name) for name in artifact_names},
    }
    atomic_json(output / "COMPLETE.json", complete)
    return complete


def validate_for_submission(
    config_path: Path, output_override: Path | None
) -> dict[str, Any]:
    config = load_yaml(config_path)
    shards, terms = validate_contract(config_path, config)
    rows = sum(pq.ParquetFile(path).metadata.num_rows for path in shards)
    if rows != int(config["input"]["expected_rows"]):
        raise ValueError(f"Parquet footer rows {rows} differ from frozen contract")
    output = output_override or resolve_path(config["output"]["directory"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing nonempty output directory: {output}")
    return {
        "status": "valid_for_submission",
        "shards": len(shards),
        "rows": rows,
        "terms": terms,
        "output": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "openlibrary/config/temporal.yaml",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        print(
            json.dumps(
                validate_for_submission(args.config.resolve(), args.output),
                sort_keys=True,
            )
        )
        return
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    result = run(args.config.resolve(), args.output, args.batch_size)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
