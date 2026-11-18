#!/usr/bin/env python3
"""Count each frozen explicit-climate phrase by year and metadata field.

The script scans the complete eligible-work corpus once, emits compact
aggregates only, and never materializes work-level matches.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

from openlibrary.statistics import grouped_binomial_trend
from openlibrary.common import (
    ROOT,
    atomic_csv,
    atomic_json,
    file_record,
    resolve_path,
    utc_now,
)

FIELDS = ("title", "subjects", "description", "any")
COLUMNS = ["year", "title", "subtitle", "subjects", "best_description"]


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def term_pattern(term: str) -> str:
    if not str(term).strip():
        raise ValueError("Explicit-climate terms must be nonempty")
    return rf"(?i)\b(?:{re.escape(str(term).casefold())})\b"


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


def field_values(
    table: pa.Table,
) -> dict[str, pa.Array | pa.ChunkedArray]:
    return {
        "title": pc.binary_join_element_wise(
            arrow_strings(table["title"]),
            arrow_strings(table["subtitle"]),
            " ",
            null_handling="replace",
        ),
        "subjects": pc.fill_null(pc.binary_join(table["subjects"], " | "), ""),
        "description": arrow_strings(table["best_description"]),
    }


def empty_counts(
    terms: Sequence[str], first_year: int, last_year: int
) -> dict[str, Any]:
    bins = last_year - first_year + 1
    return {
        "works": np.zeros(bins, dtype=np.int64),
        "available": {field: np.zeros(bins, dtype=np.int64) for field in FIELDS},
        "terms": {
            term: {field: np.zeros(bins, dtype=np.int64) for field in FIELDS}
            for term in terms
        },
        "all_terms_any": np.zeros(bins, dtype=np.int64),
    }


def accumulate_batch(
    table: pa.Table,
    *,
    terms: Sequence[str],
    first_year: int,
    last_year: int,
    counts: dict[str, Any],
) -> int:
    years = np.asarray(table["year"].to_numpy(zero_copy_only=False), dtype=np.int64)
    if not ((years >= first_year) & (years <= last_year)).all():
        raise ValueError("Eligible-work shard contains an out-of-scope year")
    indices = years - first_year
    bins = last_year - first_year + 1
    values = field_values(table)
    available = {field: string_available(value) for field, value in values.items()}
    available["any"] = np.ones(len(table), dtype=bool)
    counts["works"] += np.bincount(indices, minlength=bins).astype(np.int64)
    for field in FIELDS:
        counts["available"][field] += np.bincount(
            indices, weights=available[field].astype(np.int64), minlength=bins
        ).astype(np.int64)

    all_terms_any = np.zeros(len(table), dtype=bool)
    for term in terms:
        pattern = term_pattern(term)
        masks = {
            field: boolean_numpy(pc.match_substring_regex(value, pattern))
            for field, value in values.items()
        }
        masks["any"] = np.logical_or.reduce(list(masks.values()))
        all_terms_any |= masks["any"]
        for field in FIELDS:
            counts["terms"][term][field] += np.bincount(
                indices, weights=masks[field].astype(np.int64), minlength=bins
            ).astype(np.int64)
    counts["all_terms_any"] += np.bincount(
        indices, weights=all_terms_any.astype(np.int64), minlength=bins
    ).astype(np.int64)
    return len(table)


def percentage(numerator: int, denominator: int) -> float | None:
    return None if denominator <= 0 else 100.0 * numerator / denominator


def annual_rows(
    counts: Mapping[str, Any],
    terms: Sequence[str],
    first_year: int,
    last_year: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for offset, year in enumerate(range(first_year, last_year + 1)):
        works = int(counts["works"][offset])
        for term in terms:
            for field in FIELDS:
                available = int(counts["available"][field][offset])
                matches = int(counts["terms"][term][field][offset])
                rows.append(
                    {
                        "year": year,
                        "term": term,
                        "field": field,
                        "works": works,
                        "field_available": available,
                        "match_count": matches,
                        "percent_of_works": percentage(matches, works),
                        "percent_of_available": percentage(matches, available),
                    }
                )
    return rows


def trend_rows(
    annual: Sequence[Mapping[str, Any]],
    terms: Sequence[str],
    first_year: int,
    post_peak_first_year: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for term in terms:
        for field in FIELDS:
            denominator = (
                "field_available" if field in {"subjects", "description"} else "works"
            )
            for window, window_first_year, role in (
                ("full", first_year, "descriptive_full_series"),
                ("post_2010", post_peak_first_year, "exploratory_post_2010"),
            ):
                selected = [
                    row
                    for row in annual
                    if row["term"] == term
                    and row["field"] == field
                    and row["year"] >= window_first_year
                ]
                estimate = grouped_binomial_trend(
                    np.asarray([row["year"] for row in selected], dtype=np.int64),
                    np.asarray(
                        [row["match_count"] for row in selected], dtype=np.int64
                    ),
                    np.asarray([row[denominator] for row in selected], dtype=np.int64),
                )
                rows.append(
                    {
                        "term": term,
                        "field": field,
                        "window": window,
                        "first_year": window_first_year,
                        "denominator": denominator,
                        "role": role,
                        **estimate,
                    }
                )
    return rows


def validate_contract(config: Mapping[str, Any]) -> tuple[list[Path], list[str]]:
    if config.get("status") != "frozen_before_term_diagnostic":
        raise ValueError("Term diagnostic config is not frozen")
    input_config = config["input"]
    marker_path = Path(input_config["complete_marker"])
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if int(marker.get("eligible_works", -1)) != int(input_config["expected_rows"]):
        raise ValueError("Eligible-work marker row count differs from contract")
    if int(marker.get("buckets", -1)) != int(input_config["expected_shards"]):
        raise ValueError("Eligible-work marker shard count differs from contract")
    shards = sorted(Path(input_config["eligible_root"]).glob("bucket-*.parquet"))
    if len(shards) != int(input_config["expected_shards"]):
        raise ValueError("Eligible-work shard count differs from contract")
    lexical = config["lexical_contract"]
    terms_path = resolve_path(lexical["source_terms_file"])
    source_terms = load_yaml(terms_path)[lexical["source_group"]]
    terms = [str(term) for term in lexical["explicit_terms"]]
    if list(source_terms) != terms or len(terms) != len(set(terms)):
        raise ValueError("Term list drifted from its source group")
    if tuple(config["scope"]["fields"]) != FIELDS:
        raise ValueError("Configured fields changed")
    return shards, terms


def scan_shards(
    shards: Sequence[Path],
    *,
    terms: Sequence[str],
    first_year: int,
    last_year: int,
    batch_size: int,
) -> tuple[dict[str, Any], int]:
    counts = empty_counts(terms, first_year, last_year)
    rows = 0
    for shard in shards:
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=COLUMNS):
            rows += accumulate_batch(
                pa.Table.from_batches([batch]),
                terms=terms,
                first_year=first_year,
                last_year=last_year,
                counts=counts,
            )
    return counts, rows


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def markdown_report(
    config: Mapping[str, Any],
    annual: Sequence[Mapping[str, Any]],
    trends: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# Full-corpus term-by-term annual diagnostic v1",
        "",
        "All counts use the complete in-scope work corpus. The 2010--2025 models "
        "are explicitly exploratory because the cutoff was selected after viewing "
        "the combined annual series.",
        "",
        "| Term | Any-field matches | Full OR/decade | 2010--2025 OR/decade |",
        "|---|---:|---:|---:|",
    ]
    for term in config["lexical_contract"]["explicit_terms"]:
        matches = sum(
            int(row["match_count"])
            for row in annual
            if row["term"] == term and row["field"] == "any"
        )
        full = next(
            row
            for row in trends
            if row["term"] == term and row["field"] == "any" and row["window"] == "full"
        )
        post = next(
            row
            for row in trends
            if row["term"] == term
            and row["field"] == "any"
            and row["window"] == "post_2010"
        )

        def display(row: Mapping[str, Any]) -> str:
            if row["status"] != "estimated":
                return "n/e"
            return f"{row['odds_ratio_per_decade']:.4f}"

        lines.append(f"| {term} | {matches:,} | {display(full)} | {display(post)} |")
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            str(config["claim_guardrail"]),
            "",
        ]
    )
    return "\n".join(lines)


def run(
    config_path: Path, output_override: Path | None, batch_size: int
) -> dict[str, Any]:
    config = load_yaml(config_path)
    shards, terms = validate_contract(config)
    scope = config["scope"]
    first_year = int(scope["first_year"])
    last_year = int(scope["last_year"])
    output = output_override or resolve_path(config["output"]["directory"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    counts, rows_scanned = scan_shards(
        shards,
        terms=terms,
        first_year=first_year,
        last_year=last_year,
        batch_size=batch_size,
    )
    if rows_scanned != int(config["input"]["expected_rows"]):
        raise ValueError("Scanned row count differs from contract")
    annual = annual_rows(counts, terms, first_year, last_year)
    trends = trend_rows(
        annual,
        terms,
        first_year,
        int(scope["exploratory_post_peak_first_year"]),
    )
    annual_fields = [
        "year",
        "term",
        "field",
        "works",
        "field_available",
        "match_count",
        "percent_of_works",
        "percent_of_available",
    ]
    trend_fields = [
        "term",
        "field",
        "window",
        "first_year",
        "denominator",
        "role",
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
    atomic_csv(output / "annual_term_counts.csv", annual, annual_fields)
    atomic_csv(output / "trend_estimates.csv", trends, trend_fields)
    result = {
        "status": "complete",
        "created_utc": utc_now(),
        "source_snapshot": config["source_snapshot"],
        "rows_scanned": rows_scanned,
        "terms": terms,
        "all_terms_unique_any_matches": int(counts["all_terms_any"].sum()),
        "annual_term_counts": annual,
        "trend_estimates": trends,
        "claim_guardrail": config["claim_guardrail"],
    }
    atomic_json(output / "report.json", result)
    atomic_text(output / "README.md", markdown_report(config, annual, trends))
    provenance = {
        "created_utc": utc_now(),
        "config": {"path": str(config_path), **file_record(config_path)},
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
    artifacts = [
        "annual_term_counts.csv",
        "trend_estimates.csv",
        "report.json",
        "README.md",
        "provenance.json",
    ]
    complete = {
        "status": "complete",
        "created_utc": utc_now(),
        "rows_scanned": rows_scanned,
        "source_snapshot": config["source_snapshot"],
        "artifacts": {name: file_record(output / name) for name in artifacts},
    }
    atomic_json(output / "COMPLETE.json", complete)
    return complete


def validate_for_submission(
    config_path: Path, output_override: Path | None
) -> dict[str, Any]:
    config = load_yaml(config_path)
    shards, terms = validate_contract(config)
    rows = sum(pq.ParquetFile(path).metadata.num_rows for path in shards)
    if rows != int(config["input"]["expected_rows"]):
        raise ValueError("Parquet footer rows differ from contract")
    schema_names = set(pq.ParquetFile(shards[0]).schema_arrow.names)
    if not set(COLUMNS).issubset(schema_names):
        raise ValueError("Eligible-work schema is missing required columns")
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


def smoke(config_path: Path, rows: int) -> dict[str, Any]:
    config = load_yaml(config_path)
    shards, terms = validate_contract(config)
    scope = config["scope"]
    batch = next(
        pq.ParquetFile(shards[0]).iter_batches(batch_size=rows, columns=COLUMNS)
    )
    table = pa.Table.from_batches([batch]).slice(0, rows)
    counts = empty_counts(terms, int(scope["first_year"]), int(scope["last_year"]))
    scanned = accumulate_batch(
        table,
        terms=terms,
        first_year=int(scope["first_year"]),
        last_year=int(scope["last_year"]),
        counts=counts,
    )
    return {
        "status": "smoke_complete",
        "rows": scanned,
        "terms": len(terms),
        "all_terms_unique_any_matches": int(counts["all_terms_any"].sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "openlibrary/config/vocabulary.yaml",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--smoke-rows", type=int)
    args = parser.parse_args()
    config_path = args.config.resolve()
    if args.validate_only:
        print(
            json.dumps(
                validate_for_submission(config_path, args.output), sort_keys=True
            )
        )
        return
    if args.smoke_rows is not None:
        if args.smoke_rows <= 0:
            parser.error("--smoke-rows must be positive")
        print(json.dumps(smoke(config_path, args.smoke_rows), sort_keys=True))
        return
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    print(json.dumps(run(config_path, args.output, args.batch_size), sort_keys=True))


if __name__ == "__main__":
    main()
