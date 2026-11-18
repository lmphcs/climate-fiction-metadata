#!/usr/bin/env python3
"""Build the frozen, evaluation-only CROW held-out annotation packets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

STRATA = (
    "explicit_only",
    "motif_only",
    "semantic_only",
    "route_overlap",
    "outside_pool",
)
SELECTION_TIERS = (
    "fiction_clean",
    "fiction_conflict",
    "lexical_no_fiction_signal",
)
ELIGIBLE_COLUMNS = (
    "work_key",
    "title",
    "subtitle",
    "authors",
    "year",
    "subjects",
    "best_description",
    "fiction_likely",
    "fiction_signals",
    "nonfiction_signals",
    "fiction_signal_conflict",
    "split",
)
BLINDED_FIELDS = (
    "annotation_id",
    "title",
    "subtitle",
    "authors",
    "first_publish_year",
    "subjects",
    "description",
)
KEY_FIELDS = BLINDED_FIELDS + (
    "retest_annotation_id",
    "retest_selected",
    "work_key",
    "evaluation_stratum",
    "split",
    "fiction_likely",
    "fiction_signal_conflict",
    "explicit_matches",
    "high_specific_matches",
    "contextual_families",
    "contextual_matches",
    "environmental_context_matches",
    "broad_hard_negative_matches",
    "explicit_route",
    "motif_route",
    "semantic_rank",
    "semantic_score",
    "best_query_id",
    "sampling_tier",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def file_record(path: Path) -> Dict[str, Any]:
    return {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def load_yaml(path: Path) -> Dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected YAML mapping: {path}")
    return value


def read_exclusions(paths: Sequence[Path]) -> tuple[set[str], List[Dict[str, Any]]]:
    keys: set[str] = set()
    records: List[Dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise RuntimeError(f"Exclusion key missing: {path}")
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        before = len(keys)
        keys.update(row["work_key"] for row in rows if row.get("work_key"))
        records.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "rows": len(rows),
                "new_unique_work_keys": len(keys) - before,
            }
        )
    return keys, records


@lru_cache(maxsize=None)
def phrase_pattern(term: str) -> re.Pattern[str]:
    return re.compile(r"(?<!\w)" + re.escape(term.casefold()) + r"(?!\w)")


def matches(text: str, terms: Sequence[str]) -> List[str]:
    return sorted(term for term in terms if phrase_pattern(term).search(text))


def lexical_features(text: str, terms: Mapping[str, Any]) -> Dict[str, Any]:
    folded = text.casefold()
    explicit = matches(folded, terms["explicit_climate"])
    high = matches(folded, terms["high_specific_motif"])
    family_matches: Dict[str, List[str]] = {}
    for family, values in terms["contextual_motif_families"].items():
        found = matches(folded, values)
        if found:
            family_matches[family] = found
    contextual = sorted(value for values in family_matches.values() for value in values)
    environmental = matches(folded, terms["environmental_genre_context"])
    broad = matches(folded, terms["broad_hard_negative"])
    return {
        "explicit_matches": explicit,
        "high_specific_matches": high,
        "contextual_families": sorted(family_matches),
        "contextual_matches": contextual,
        "environmental_context_matches": environmental,
        "broad_hard_negative_matches": broad,
        "explicit_route": bool(explicit),
        "motif_route": bool(high) or len(family_matches) >= 2,
    }


def list_text(value: Any, separator: str = " | ") -> str:
    if not value:
        return ""
    if isinstance(value, list):
        return separator.join(str(item) for item in value if item)
    return str(value)


def document_text(row: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    document = config["document"]
    title = " ".join(
        value for value in (row.get("title"), row.get("subtitle")) if value
    )
    description = str(row.get("best_description") or "")[
        : int(document["max_description_characters"])
    ]
    subjects = list(row.get("subjects") or [])[: int(document["max_subjects"])]
    parts = {
        "title": f"Title: {title}",
        "description": f"Description: {description}",
        "subjects": f"Subjects: {list_text(subjects, '; ')}",
    }
    body = "\n".join(parts[name] for name in document["order"] if parts[name])
    return str(document["prefix"]) + body


def scan_candidates(
    eligible_dir: Path,
    config: Mapping[str, Any],
    terms: Mapping[str, Any],
    exclusions: set[str],
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    counters: Counter[str] = Counter()
    candidates: List[Dict[str, Any]] = []
    paths = sorted(eligible_dir.glob("bucket-*.parquet"))
    if not paths:
        raise RuntimeError(f"No eligible Parquet buckets: {eligible_dir}")

    for path in paths:
        parquet = pq.ParquetFile(path)
        missing = set(ELIGIBLE_COLUMNS) - set(parquet.schema_arrow.names)
        if missing:
            raise RuntimeError(f"Missing columns in {path}: {sorted(missing)}")
        for batch in parquet.iter_batches(batch_size=20_000, columns=ELIGIBLE_COLUMNS):
            for row in batch.to_pylist():
                counters["eligible_seen"] += 1
                if row["split"] != config["split"]:
                    counters["non_target_split_skipped_before_scoring"] += 1
                    continue
                counters["target_split_seen"] += 1
                if not row["best_description"]:
                    counters["target_without_description_skipped"] += 1
                    continue
                if row["work_key"] in exclusions:
                    counters["excluded_work_skipped"] += 1
                    continue

                metadata = "\n".join(
                    (
                        str(row.get("title") or ""),
                        str(row.get("subtitle") or ""),
                        list_text(row.get("subjects")),
                        str(row.get("best_description") or ""),
                    )
                )
                features = lexical_features(metadata, terms)
                if not (
                    row["fiction_likely"]
                    or features["explicit_route"]
                    or features["motif_route"]
                ):
                    counters["target_outside_candidate_union_skipped"] += 1
                    continue

                candidate = dict(row)
                candidate.update(features)
                candidate["model_text"] = document_text(row, config)
                candidates.append(candidate)
                counters["candidates_for_embedding"] += 1
                if row["fiction_likely"]:
                    counters["candidates_with_fiction_signal"] += 1
                else:
                    counters["lexical_candidates_without_fiction_signal"] += 1
                    if features["explicit_route"]:
                        counters["explicit_candidates_without_fiction_signal"] += 1
                    if features["motif_route"]:
                        counters["motif_candidates_without_fiction_signal"] += 1

    if not candidates:
        raise RuntimeError("No held-out candidates survived evaluation eligibility")
    if counters["non_target_split_skipped_before_scoring"] == 0:
        raise RuntimeError("Leakage guard failed: no non-target rows were skipped")
    return candidates, dict(counters)


def ensure_model(
    model_id: str,
    revision: str,
    model_dir: Path,
    manifest_path: Path,
) -> Dict[str, Any]:
    from huggingface_hub import snapshot_download

    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=model_dir,
        ignore_patterns=[
            "*.bin",
            "*.onnx",
            "onnx/*",
            "openvino/*",
            "*.h5",
            "*.msgpack",
            "rust_model.ot",
        ],
    )
    files = sorted(path for path in model_dir.rglob("*") if path.is_file())
    if not any(path.name == "model.safetensors" for path in files):
        raise RuntimeError("Pinned model snapshot lacks model.safetensors")
    pinned = json.loads(
        (Path(__file__).resolve().parents[1] / "sources.json").read_text()
    )["models"]["e5"]
    if (model_id, revision) == (pinned["model_id"], pinned["revision"]):
        for name, expected in pinned["files"].items():
            actual = file_record(model_dir / name)
            if any(actual[key] != expected[key] for key in ("sha256", "size_bytes")):
                raise RuntimeError(f"Pinned E5 payload mismatch: {name}")
    manifest = {
        "model_id": model_id,
        "revision": revision,
        "downloaded_utc": utc_now(),
        "files": {
            str(path.relative_to(model_dir)): file_record(path) for path in files
        },
    }
    write_json(manifest_path, manifest)
    return manifest


def encode_candidates(
    candidates: List[Dict[str, Any]],
    config: Mapping[str, Any],
    model_dir: Path,
) -> None:
    from sentence_transformers import SentenceTransformer

    model_config = config["model"]
    model = SentenceTransformer(str(model_dir), device="cpu")
    model.max_seq_length = int(model_config["max_sequence_length"])
    query_ids = [item["id"] for item in config["queries"]]
    query_texts = [item["text"] for item in config["queries"]]
    query_embeddings = model.encode(
        query_texts,
        batch_size=len(query_texts),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    batch_size = int(model_config["batch_size"])
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        embeddings = model.encode(
            [row["model_text"] for row in batch],
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        scores = np.asarray(embeddings) @ np.asarray(query_embeddings).T
        best = scores.argmax(axis=1)
        for row, row_scores, best_index in zip(batch, scores, best):
            row["semantic_score"] = float(row_scores[best_index])
            row["best_query_id"] = query_ids[int(best_index)]
            row["query_scores"] = {
                query_id: float(score) for query_id, score in zip(query_ids, row_scores)
            }


def assign_semantic_ranks(candidates: List[Dict[str, Any]]) -> None:
    ordered = sorted(
        candidates,
        key=lambda row: (-row["semantic_score"], row["work_key"]),
    )
    for rank, row in enumerate(ordered, start=1):
        row["semantic_rank"] = rank


def stratum_for(row: Mapping[str, Any], top_k: int) -> str:
    semantic = int(row["semantic_rank"]) <= top_k
    explicit = bool(row["explicit_route"])
    motif = bool(row["motif_route"])
    if semantic and (explicit or motif):
        return "route_overlap"
    if semantic:
        return "semantic_only"
    if explicit and not motif:
        return "explicit_only"
    if motif and not explicit:
        return "motif_only"
    if not explicit and not motif:
        return "outside_pool"
    return ""


def stable_priority(seed: int, stratum: str, work_key: str) -> str:
    return hashlib.sha256(f"{seed}:{stratum}:{work_key}".encode()).hexdigest()


def select_packet(
    candidates: Sequence[Dict[str, Any]],
    config: Mapping[str, Any],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    top_k = int(config["selection"]["semantic_top_k"])
    per_stratum = int(config["per_stratum"])
    seed = int(config["seed"])
    tier_order = tuple(config["selection"]["prefer_tiers"])
    pools: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        name: {tier: [] for tier in tier_order} for name in STRATA
    }
    for row in candidates:
        stratum = stratum_for(row, top_k)
        if not stratum:
            continue
        if row["fiction_likely"]:
            tier = (
                "fiction_conflict"
                if row["fiction_signal_conflict"]
                else "fiction_clean"
            )
        else:
            if not (row["explicit_route"] or row["motif_route"]):
                raise RuntimeError("No-fiction-signal candidate lacks a lexical route")
            tier = "lexical_no_fiction_signal"
        pools[stratum][tier].append(row)

    selected: List[Dict[str, Any]] = []
    diagnostics: Dict[str, Any] = {"pool_counts": {}, "selected_tiers": {}}
    for stratum in STRATA:
        for tier in tier_order:
            pools[stratum][tier].sort(
                key=lambda row: stable_priority(seed, stratum, row["work_key"])
            )
        diagnostics["pool_counts"][stratum] = {
            tier: len(pools[stratum][tier]) for tier in tier_order
        }
        chosen: List[tuple[Dict[str, Any], str]] = []
        for tier in tier_order:
            needed = per_stratum - len(chosen)
            if needed <= 0:
                break
            chosen.extend((row, tier) for row in pools[stratum][tier][:needed])
        if len(chosen) != per_stratum:
            found = ", ".join(
                f"{tier}={len(pools[stratum][tier])}" for tier in tier_order
            )
            raise RuntimeError(
                f"Evaluation stratum {stratum} needs {per_stratum}; found {found}"
            )
        diagnostics["selected_tiers"][stratum] = dict(
            Counter(tier for _, tier in chosen)
        )
        selected.extend(
            dict(row, evaluation_stratum=stratum, sampling_tier=tier)
            for row, tier in chosen
        )

    selected.sort(
        key=lambda row: stable_priority(seed, "blinded-order", row["work_key"])
    )
    for index, row in enumerate(selected, start=1):
        prefix = str(config["evaluation"]["primary_id_prefix"])
        row["annotation_id"] = f"{prefix}-{index:03d}"
    return selected, diagnostics


def select_retest(
    selected: Sequence[Dict[str, Any]],
    config: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Choose a balanced delayed retest subset without exposing the mapping."""
    seed = int(config["seed"])
    per_stratum = int(config["retest_per_stratum"])
    chosen: List[Dict[str, Any]] = []
    for stratum in STRATA:
        pool = [row for row in selected if row["evaluation_stratum"] == stratum]
        pool.sort(
            key=lambda row: stable_priority(seed, f"retest:{stratum}", row["work_key"])
        )
        if len(pool) < per_stratum:
            raise RuntimeError(
                f"Retest stratum {stratum} needs {per_stratum}; found {len(pool)}"
            )
        chosen.extend(pool[:per_stratum])
    chosen.sort(key=lambda row: stable_priority(seed, "retest-order", row["work_key"]))
    prefix = str(config["evaluation"]["retest_id_prefix"])
    for index, row in enumerate(chosen, start=1):
        row["retest_annotation_id"] = f"{prefix}-{index:03d}"
    return chosen


def serialise_list(value: Any) -> str:
    return " | ".join(value) if isinstance(value, list) else str(value or "")


def visible_row(source: Mapping[str, Any], annotation_id: str) -> Dict[str, Any]:
    return {
        "annotation_id": annotation_id,
        "title": source.get("title") or "",
        "subtitle": source.get("subtitle") or "",
        "authors": serialise_list(source.get("authors")),
        "first_publish_year": source.get("year") or "",
        "subjects": serialise_list(source.get("subjects")),
        "description": source.get("best_description") or "",
    }


def packet_rows(
    selected: Sequence[Mapping[str, Any]],
    retest: Sequence[Mapping[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    blinded: List[Dict[str, Any]] = []
    key: List[Dict[str, Any]] = []
    for source in selected:
        visible = visible_row(source, str(source["annotation_id"]))
        hidden = dict(visible)
        hidden.update(
            {
                "retest_annotation_id": source.get("retest_annotation_id") or "",
                "retest_selected": bool(source.get("retest_annotation_id")),
                "work_key": source["work_key"],
                "evaluation_stratum": source["evaluation_stratum"],
                "split": source["split"],
                "fiction_likely": source["fiction_likely"],
                "fiction_signal_conflict": source["fiction_signal_conflict"],
                "explicit_matches": serialise_list(source["explicit_matches"]),
                "high_specific_matches": serialise_list(
                    source["high_specific_matches"]
                ),
                "contextual_families": serialise_list(source["contextual_families"]),
                "contextual_matches": serialise_list(source["contextual_matches"]),
                "environmental_context_matches": serialise_list(
                    source["environmental_context_matches"]
                ),
                "broad_hard_negative_matches": serialise_list(
                    source["broad_hard_negative_matches"]
                ),
                "explicit_route": source["explicit_route"],
                "motif_route": source["motif_route"],
                "semantic_rank": source["semantic_rank"],
                "semantic_score": f"{source['semantic_score']:.8f}",
                "best_query_id": source["best_query_id"],
                "sampling_tier": source["sampling_tier"],
            }
        )
        blinded.append(visible)
        key.append(hidden)
    retest_blinded = [
        visible_row(source, str(source["retest_annotation_id"])) for source in retest
    ]
    return blinded, key, retest_blinded


def write_scores(path: Path, candidates: Sequence[Mapping[str, Any]]) -> None:
    rows = []
    for row in candidates:
        item = {
            "work_key": row["work_key"],
            "split": row["split"],
            "semantic_rank": row["semantic_rank"],
            "semantic_score": row["semantic_score"],
            "best_query_id": row["best_query_id"],
            "explicit_route": row["explicit_route"],
            "motif_route": row["motif_route"],
            "fiction_likely": row["fiction_likely"],
            "fiction_signal_conflict": row["fiction_signal_conflict"],
        }
        item.update(
            {f"query_{key}": value for key, value in row["query_scores"].items()}
        )
        rows.append(item)
    rows.sort(key=lambda row: row["semantic_rank"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eligible-dir", type=Path, required=True)
    parser.add_argument("--semantic-config", type=Path, required=True)
    parser.add_argument("--terms", type=Path, required=True)
    parser.add_argument("--codebook", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--environment-freeze", type=Path, required=True)
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--score-output", type=Path, required=True)
    parser.add_argument("--packet-dir", type=Path, required=True)
    parser.add_argument("--exclude-key", type=Path, action="append", default=[])
    parser.add_argument("--overwrite-incomplete", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.semantic_config)
    terms = load_yaml(args.terms)
    if config.get("status") != "frozen" or config.get("split") != "evaluation":
        raise RuntimeError("Evaluation config must be frozen and target evaluation")
    if terms.get("status") != "frozen":
        raise RuntimeError("Retrieval terms must be the frozen version")
    if not args.codebook.is_file():
        raise RuntimeError(f"Frozen codebook missing: {args.codebook}")
    if tuple(config["selection"]["strata"]) != STRATA:
        raise RuntimeError("Semantic config strata/order do not match implementation")
    if tuple(config["selection"]["candidate_union_any"]) != (
        "fiction_likely",
        "explicit_route",
        "motif_route",
    ):
        raise RuntimeError("Candidate-union config does not match implementation")
    if tuple(config["selection"]["prefer_tiers"]) != SELECTION_TIERS:
        raise RuntimeError("Selection-tier config does not match implementation")
    marker_path = args.eligible_dir / "COMPLETE.json"
    if not marker_path.is_file():
        raise RuntimeError(f"Eligible completion marker missing: {marker_path}")
    exclusions, exclusion_records = read_exclusions(args.exclude_key)

    if args.packet_dir.exists():
        complete = args.packet_dir / "COMPLETE.json"
        if complete.is_file():
            print(f"Held-out evaluation packet already complete: {args.packet_dir}")
            return 0
        if not args.overwrite_incomplete:
            raise RuntimeError(f"Incomplete packet exists: {args.packet_dir}")
        shutil.rmtree(args.packet_dir)
    temporary = args.packet_dir.parent / f".{args.packet_dir.name}.inprogress"
    if temporary.exists():
        if not args.overwrite_incomplete:
            raise RuntimeError(f"Incomplete temporary packet exists: {temporary}")
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    model = config["model"]
    model_manifest = ensure_model(
        model["id"], model["revision"], args.model_dir, args.model_manifest
    )
    candidates, counters = scan_candidates(args.eligible_dir, config, terms, exclusions)
    encode_candidates(candidates, config, args.model_dir)
    assign_semantic_ranks(candidates)
    write_scores(args.score_output, candidates)
    selected, selection_diagnostics = select_packet(candidates, config)
    retest = select_retest(selected, config)
    blinded, key, retest_blinded = packet_rows(selected, retest)

    expected = len(STRATA) * int(config["per_stratum"])
    expected_retest = len(STRATA) * int(config["retest_per_stratum"])
    if expected != int(config["evaluation"]["unique_records"]):
        raise RuntimeError("Config unique-record count does not match strata")
    if expected_retest != int(config["evaluation"]["retest_records"]):
        raise RuntimeError("Config retest count does not match strata")
    if len(blinded) != expected or len({row["work_key"] for row in key}) != expected:
        raise RuntimeError("Evaluation size/uniqueness validation failed")
    if len(retest_blinded) != expected_retest:
        raise RuntimeError("Evaluation retest-size validation failed")
    if {row["split"] for row in key} != {"evaluation"}:
        raise RuntimeError("Evaluation leakage guard failed")
    if {row["work_key"] for row in key} & exclusions:
        raise RuntimeError("Evaluation overlaps an excluded work")

    blinded_name = str(config["evaluation"]["primary_filename"])
    retest_name = str(config["evaluation"]["retest_filename"])
    key_name = str(config["evaluation"]["key_filename"])
    write_csv(temporary / blinded_name, blinded, BLINDED_FIELDS)
    write_csv(temporary / retest_name, retest_blinded, BLINDED_FIELDS)
    write_csv(temporary / key_name, key, KEY_FIELDS)
    diagnostics = {
        "status": "complete",
        "created_utc": utc_now(),
        "scan": counters,
        "selection": selection_diagnostics,
        "selected_strata": dict(Counter(row["evaluation_stratum"] for row in key)),
        "retest_strata": dict(Counter(row["evaluation_stratum"] for row in retest)),
        "excluded_unique_work_keys": len(exclusions),
        "development_rows_scored": 0,
        "evaluation_rows_scored": len(candidates),
    }
    write_json(temporary / "diagnostics.json", diagnostics)
    provenance = {
        "generator": Path(__file__).name,
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "semantic_config": {
            "path": str(args.semantic_config.resolve()),
            "sha256": sha256_file(args.semantic_config),
        },
        "retrieval_terms": {
            "path": str(args.terms.resolve()),
            "sha256": sha256_file(args.terms),
        },
        "codebook": {
            "path": str(args.codebook.resolve()),
            "sha256": sha256_file(args.codebook),
            "version": str(config["codebook_version"]),
        },
        "eligible_marker": {
            "path": str(marker_path.resolve()),
            "sha256": sha256_file(marker_path),
        },
        "exclusions": exclusion_records,
        "model": model_manifest,
        "score_output": {
            "path": str(args.score_output.resolve()),
            **file_record(args.score_output),
        },
        "software_environment": {
            "freeze_path": str(args.environment_freeze.resolve()),
            "freeze_sha256": sha256_file(args.environment_freeze),
            "requirements_path": str(args.requirements.resolve()),
            "requirements_sha256": sha256_file(args.requirements),
        },
    }
    write_json(temporary / "provenance.json", provenance)
    readme = (
        "# CROW held-out evaluation v1\n\n"
        "The primary packet contains 100 unique held-out records: 20 each from "
        "explicit-only, motif-only, semantic-only, route-overlap, and "
        "outside-pool strata. The separate 10-record retest packet contains "
        "two blinded repeats per stratum and must not be opened or annotated "
        "until 7--10 days after primary annotation. Keep the key and model "
        "scores hidden until both annotation phases are complete. Motif is a "
        "diagnostic route, not a required success condition.\n"
    )
    (temporary / "README.md").write_text(readme, encoding="utf-8")
    packet_files = {
        name: file_record(temporary / name)
        for name in (
            blinded_name,
            retest_name,
            key_name,
            "diagnostics.json",
            "provenance.json",
            "README.md",
        )
    }
    write_json(
        temporary / "COMPLETE.json",
        {
            "status": "complete",
            "created_utc": utc_now(),
            "primary_rows": expected,
            "retest_rows": expected_retest,
            "unique_work_rows": expected,
            "packet_files": packet_files,
        },
    )
    os.replace(temporary, args.packet_dir)
    print(json.dumps(diagnostics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
