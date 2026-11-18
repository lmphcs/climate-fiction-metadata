#!/usr/bin/env python3
"""Tiny source-to-evaluation check; substitutes a test encoder, never downloads models."""

from pathlib import Path
from unittest.mock import patch
import contextlib
import gzip
import importlib
import io
import json
import sys
import tempfile
import types

import numpy as np
import yaml

import full


def run(work):
    from openlibrary.aggregate import stable_split
    from openlibrary import retrieval

    work.mkdir(parents=True, exist_ok=True)
    if any(work.iterdir()):
        raise FileExistsError("Use an empty smoke-test directory")
    payload = work / "payload"
    payload.mkdir()
    references = work / "references"
    config = json.loads((full.ROOT / "pipeline.json").read_text())
    config.update(references=str(references), verify_paper=False, buckets=2)
    sources = {"openlibrary": {"snapshot": "2026-06-30", "files": []}}
    works, editions, truth = [], [], {}
    next_id = 10001

    def add(group, description, evaluation=True, english=True, fiction=True):
        nonlocal next_id
        while True:
            key = f"/works/OL{next_id}W"
            next_id += 1
            if (stable_split(key)[1] == "evaluation") == evaluation:
                break
        title = f"Fixture {group} {next_id}"
        works.append(
            dict(
                key=key,
                title=title,
                description=description,
                first_publish_date="2000",
                authors=[{"author": {"key": "/authors/OL1A"}}],
                subjects=["Fiction"] if fiction else ["Chess"],
            )
        )
        editions.append(
            dict(
                key=f"/books/OL{next_id}M",
                title=title,
                works=[{"key": key}],
                publish_date="2000",
                languages=[{"key": "/languages/eng" if english else "/languages/fre"}],
            )
        )
        truth[key] = group
        return key

    descriptions = {
        "explicit": "A novel about climate change.",
        "motif": "A scholarly study of geoengineering and literary criticism.",
        "semantic": "Literary criticism about novels on a permanently hotter planet.",
        "overlap": "A novel about climate change and geoengineering.",
        "outside": "A novel about chess players and their family letters.",
    }
    for group, description in descriptions.items():
        for _ in range(2):
            add(group, description)
    add("development", descriptions["outside"], evaluation=False)
    add("french", descriptions["outside"], english=False)
    add("missing_description", "")
    add("no_cue", "A reference manual on chess openings.", fiction=False)
    excluded = add("excluded", descriptions["explicit"])

    for kind, records in [
        ("works", works),
        ("editions", editions),
        ("authors", [dict(key="/authors/OL1A", name="Fixture Author")]),
    ]:
        path = payload / f"ol_dump_{kind}_fixture.txt.gz"
        with path.open("wb") as raw, gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, mtime=0
        ) as output:
            for row in records:
                output.write(
                    (
                        f'/type/{kind}\t{row["key"]}\t1\t2026-06-30\t'
                        + json.dumps(row)
                        + "\n"
                    ).encode()
                )
        sources["openlibrary"]["files"].append(
            dict(
                filename=path.name,
                url=path.as_uri(),
                size_bytes=path.stat().st_size,
                md5=full.digest(path, "md5"),
            )
        )
    source_path = work / "config/sources.json"
    full.json_write(source_path, sources)
    config["sources"] = str(source_path)
    cfg = yaml.safe_load((full.ROOT / config["evaluation"]).read_text())
    cfg.update(per_stratum=1, retest_per_stratum=1)
    cfg["selection"]["semantic_top_k"] = 4
    cfg["evaluation"].update(unique_records=5, retest_records=5)
    config_path = work / "config/evaluation.yaml"
    full.yaml_write(config_path, cfg)
    config["evaluation"] = str(config_path)
    full.csv_write(
        references / "openlibrary/data/exclusions.csv", [dict(work_key=excluded)]
    )

    calls = []

    class TestEncoder:
        def __init__(self, model, device):
            assert device == "cpu"
            self.max_seq_length = None

        def encode(self, texts, **kwargs):
            assert self.max_seq_length == 512
            assert kwargs["normalize_embeddings"] is True
            assert kwargs["convert_to_numpy"] is True
            calls.append(len(texts))
            if texts[0].startswith("query: "):
                assert len(texts) == 4 and all(t.startswith("query: ") for t in texts)
                return np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
            assert all(t.startswith("passage: Title: ") for t in texts)
            return np.array(
                [
                    (
                        [1.0, 0.0]
                        if "Fixture semantic " in t
                        else (
                            [0.0, 1.0]
                            if "Fixture overlap " in t
                            else [2**-0.5, 2**-0.5]
                        )
                    )
                    for t in texts
                ]
            )

    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = TestEncoder

    def model_stub(model_id, revision, folder, manifest):
        folder.mkdir(parents=True, exist_ok=True)
        record = dict(model_id=model_id, revision=revision, test_encoder=True)
        full.json_write(manifest, record)
        return record

    def in_process_call(module, *args):
        # Exercise the actual module argument parsers and stage functions, while
        # keeping the tiny encoder double inside this process.
        with patch.object(sys, "argv", [module, *map(str, args)]):
            result = importlib.import_module(module).main()
        assert result in (None, 0)

    def after(stage, work, config):
        if stage == "prepare":
            marker = json.loads(
                (work / "openlibrary/eligible/COMPLETE.json").read_text()
            )
            assert marker["eligible_works"] == 14
        if stage == "retrieve":
            diagnostics = json.loads(
                (work / "openlibrary/packet/diagnostics.json").read_text()
            )
            assert diagnostics["scan"]["candidates_for_embedding"] == 10
            assert diagnostics["scan"]["excluded_work_skipped"] == 1
            assert diagnostics["scan"]["target_without_description_skipped"] == 1
            rows = full.read(work / "openlibrary/packet/evaluation_100_key.csv")
            for row in rows:
                group = truth[row["work_key"]]
                assert (
                    row["evaluation_stratum"]
                    == dict(
                        explicit="explicit_only",
                        motif="motif_only",
                        semantic="semantic_only",
                        overlap="route_overlap",
                        outside="outside_pool",
                    )[group]
                )
                fiction = "NONFICTION" if group in {"motif", "semantic"} else "FICTION"
                relevance = (
                    ""
                    if fiction == "NONFICTION"
                    else "NONE" if group == "outside" else "CLIMATE_EXPLICIT"
                )
                row.update(
                    fiction_status=fiction,
                    environment_climate_relevance=relevance,
                    derived_status=(
                        "POSITIVE" if group in {"explicit", "overlap"} else "NEGATIVE"
                    ),
                )
            full.csv_write(references / "openlibrary/data/evaluation.csv", rows)

    with patch.object(full, "call", in_process_call), patch.object(
        retrieval, "ensure_model", model_stub
    ), patch.dict(sys.modules, {"sentence_transformers": fake}):
        full.run_pipeline(work, config, after=after)
    assert calls == [4, 10]
    result = json.loads((work / "results/evaluation.json").read_text())
    assert result == dict(
        records=5,
        strata={s: 1 for s in retrieval.STRATA},
        judgments={"NEGATIVE": 3, "POSITIVE": 2},
    )
    for name, file, count_key, expected in [
        ("temporal", "annual_counts.csv", "explicit_count", 5),
        ("vocabulary", "annual_term_counts.csv", "match_count", 5),
    ]:
        rows = full.read(work / "openlibrary/data" / name / file)
        selected = [
            r
            for r in rows
            if r["field"] == "any"
            and (
                r.get("population") == "all_works"
                if name == "temporal"
                else r["term"] == "climate change"
            )
        ]
        assert sum(int(r["works"]) for r in selected) == 14
        assert sum(int(r[count_key]) for r in selected) == expected
    # Verify that changed source evidence cannot inherit the frozen labels.
    labels_path = references / "openlibrary/data/evaluation.csv"
    labels = full.read(labels_path)
    labels[0]["description"] = "Changed evidence"
    full.csv_write(labels_path, labels)
    try:
        full.evaluate(work, config)
    except ValueError as exc:
        assert "metadata or assignment changed" in str(exc)
    else:
        raise AssertionError("Changed evidence inherited old annotations")
    # Exercise the documented new-sample annotation path separately.
    labels[0]["description"] = full.read(
        work / "openlibrary/packet/evaluation_100_key.csv"
    )[0]["description"]
    full.csv_write(labels_path, labels)
    full.evaluate(work, dict(config, annotations=str(labels_path)))
    return dict(
        status="passed",
        synthetic=True,
        source_works=15,
        eligible_works=14,
        candidates=10,
        evaluation_records=5,
        strata=5,
        positive=2,
        stages=full.STAGES,
        real_model_inference=False,
        encoder="deterministic test double; actual prefixes, normalisation arguments, dot products and sampling exercised",
        changed_metadata_rejected=True,
        custom_annotations_checked=True,
    )


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="crow-ol-smoke-") as directory:
        with contextlib.redirect_stdout(io.StringIO()):
            summary = run(Path(directory))
    print(json.dumps(summary, indent=2))
