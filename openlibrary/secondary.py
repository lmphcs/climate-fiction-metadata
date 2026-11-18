"""Reconstruct and rerun the published secondary development benchmark on a GPU."""

from pathlib import Path
import hashlib
import json
import heapq
import csv
import yaml
import pyarrow.parquet as pq
from openlibrary import secondary_core as core
from openlibrary.retrieval import lexical_features, list_text
from openlibrary.common import atomic_csv, atomic_json, file_record

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def priority(seed, namespace, value):
    return hashlib.sha256(f"{seed}:{namespace}:{value}".encode()).hexdigest()


def visible(row, policy, identity):
    result = dict(identity)
    result.update(
        title=str(row.get("title") or "")[: policy["max_title_characters"]],
        subtitle=str(row.get("subtitle") or "")[: policy["max_subtitle_characters"]],
        authors=" | ".join((row.get("authors") or [])[: policy["max_authors"]]),
        subjects=" | ".join((row.get("subjects") or [])[: policy["max_subjects"]]),
        year=str(row.get("year") or ""),
        description=str(row.get("best_description") or "")[
            : policy["max_description_characters"]
        ],
    )
    return result


def prepare(work, config):
    frozen = read(ROOT / "openlibrary/data/secondary.csv")
    labelled = {r["work_key"]: r for r in frozen if r["gold_fiction_status"]}
    exclusions = {r["work_key"] for r in read(ROOT / "openlibrary/data/exclusions.csv")}
    terms = yaml.safe_load((ROOT / "openlibrary/config/terms.yaml").read_text())
    candidates = {}
    known = {}
    columns = [
        "work_key",
        "split",
        "fiction_likely",
        "title",
        "subtitle",
        "authors",
        "year",
        "subjects",
        "best_description",
    ]
    for p in sorted((work / "openlibrary/eligible").glob("bucket-*.parquet")):
        for batch in pq.ParquetFile(p).iter_batches(batch_size=20000, columns=columns):
            for r in batch.to_pylist():
                key = r["work_key"]
                if r["split"] != "development" or not r["best_description"]:
                    continue
                if key in labelled:
                    known[key] = r
                if key in exclusions:
                    continue
                metadata = "\n".join(
                    [
                        r.get("title") or "",
                        r.get("subtitle") or "",
                        list_text(r.get("subjects")),
                        r["best_description"],
                    ]
                )
                f = lexical_features(metadata, terms)
                if r["fiction_likely"] or f["explicit_route"] or f["motif_route"]:
                    candidates[key] = r
    keys = heapq.nsmallest(
        config["input"]["development_sample_size"],
        candidates,
        key=lambda k: priority(config["seed"], "benchmark", k),
    )
    frozen_unlabelled = {r["work_key"] for r in frozen if not r["gold_fiction_status"]}
    if set(keys) != frozen_unlabelled or set(known) != set(labelled):
        raise ValueError(
            "Secondary sample changed; freeze a new benchmark and human labels rather than reusing these results."
        )
    identities = {
        r["work_key"]: {
            k: r[k] for k in ["record_id", "work_key", "split", "source_group"]
        }
        for r in frozen
    }
    rows = [
        visible(
            (known if k in known else candidates)[k], config["input"], identities[k]
        )
        for k in keys + list(known)
    ]
    rows.sort(key=lambda r: priority(config["seed"], "benchmark-order", r["record_id"]))
    by_key = {r["work_key"]: r for r in frozen}
    for r in rows:
        value = json.dumps(
            {k: r[k] for k in core.VISIBLE_FIELDS}, sort_keys=True, ensure_ascii=False
        ).encode()
        if (
            hashlib.sha256(value).hexdigest()
            != by_key[r["work_key"]]["metadata_sha256"]
        ):
            raise ValueError(
                "Secondary displayed metadata differs from the frozen benchmark: "
                + r["work_key"]
            )
    return rows


def run(work):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from vllm import LLM

    config = yaml.safe_load((ROOT / "openlibrary/config/secondary.yaml").read_text())
    model = config["model"]
    rows = prepare(work, config)
    folder = work / "models/secondary"
    snapshot_download(
        repo_id=model["id"],
        revision=model["revision"],
        local_dir=folder,
        ignore_patterns=["*.bin", "*.h5", "*.msgpack", "onnx/*", "openvino/*"],
    )
    pinned = json.loads((ROOT / "sources.json").read_text())["models"]["secondary"]
    if (model["id"], model["revision"]) == (pinned["id"], pinned["revision"]):
        for name, expected in pinned["files"].items():
            if file_record(folder / name) != expected:
                raise ValueError("Pinned model payload mismatch: " + name)
    tokenizer = AutoTokenizer.from_pretrained(
        folder, local_files_only=True, trust_remote_code=False
    )
    llm = LLM(
        model=str(folder),
        tokenizer=str(folder),
        dtype=model["dtype"],
        max_model_len=model["max_model_len"],
        seed=config["seed"],
        gpu_memory_utilization=model["gpu_memory_utilization"],
        enforce_eager=model["enforce_eager"],
        enable_prefix_caching=model["enable_prefix_caching"],
        logprobs_mode=model["logprobs_mode"],
        trust_remote_code=False,
    )
    templates = {
        k: (ROOT / config["prompts"][k]).read_text()
        for k in ["fiction", "relevance", "evidence"]
    }
    output = []
    size = config["classification"]["chunk_size"]
    for start in range(0, len(rows), size):
        batch = rows[start : start + size]
        answers = {}
        for task in ["fiction", "relevance"]:
            prompts = [
                core.chat_prompt(
                    tokenizer,
                    config["prompts"]["system"],
                    core.render_prompt(templates[task], r),
                )
                for r in batch
            ]
            core.preflight_prompt_tokens(
                tokenizer,
                prompts,
                output_tokens=1,
                maximum_model_tokens=model["max_model_len"],
                task=task,
            )
            answers[task] = core.run_choices(
                llm,
                tokenizer,
                prompts,
                tuple(config["classification"][task + "_codes"]),
                config["seed"],
                0.0,
                1,
            )
        positive = [
            i
            for i, (f, r) in enumerate(zip(answers["fiction"], answers["relevance"]))
            if f["code"] == "F"
            and r["code"] in config["classification"]["positive_relevance_codes"]
        ]
        evidence = {}
        if positive:
            selected = [batch[i] for i in positive]
            prompts = [
                core.chat_prompt(
                    tokenizer,
                    config["prompts"]["system"],
                    core.render_prompt(
                        templates["evidence"],
                        r,
                        label=config["classification"]["relevance_codes"][
                            answers["relevance"][i]["code"]
                        ],
                    ),
                )
                for i, r in zip(positive, selected)
            ]
            core.preflight_prompt_tokens(
                tokenizer,
                prompts,
                output_tokens=config["evidence"]["max_tokens"],
                maximum_model_tokens=model["max_model_len"],
                task="evidence",
            )
            results = core.run_evidence(
                llm,
                prompts,
                selected,
                config["seed"],
                0.0,
                config["evidence"]["max_tokens"],
                config["evidence"]["no_evidence_token"],
            )
            evidence = dict(zip(positive, results))
        output.extend(
            core.result_rows(
                batch, answers["fiction"], answers["relevance"], evidence, config, start
            )
        )
    frozen = {r["work_key"]: r for r in read(ROOT / "openlibrary/data/secondary.csv")}
    compact = []
    for row in output:
        final = dict(frozen[row["work_key"]])
        for key in final:
            if key in row:
                final[key] = row[key]
        compact.append(final)
    atomic_csv(work / "openlibrary/data/secondary.csv", compact, list(compact[0]))
    path = work / "openlibrary/secondary_predictions.csv"
    atomic_csv(path, output, list(output[0]))
    atomic_json(
        work / "openlibrary/secondary_statistics.json",
        dict(
            records=len(output),
            positive=sum(r["model_positive"] for r in output),
            grounded=sum(
                r["model_positive"] and r["evidence_status"] == "grounded"
                for r in output
            ),
            model=model,
        ),
    )
