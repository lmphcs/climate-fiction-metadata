#!/usr/bin/env python3
"""Compact-data checks; pipeline checks run when Arrow is installed."""

from pathlib import Path
import csv
import hashlib
import importlib
import importlib.util
import json
import tempfile
import unittest
import yaml

ROOT = Path(__file__).resolve().parent
ARROW = importlib.util.find_spec("pyarrow") is not None


class CompactChecks(unittest.TestCase):
    def test_final_decision_rule(self):
        from openlibrary.evaluation import derived_status

        self.assertEqual(
            derived_status("FICTION", "IMPLICIT_OR_ENVIRONMENTAL"), "POSITIVE"
        )
        self.assertEqual(derived_status("FICTION", "CLIMATE_EXPLICIT"), "POSITIVE")
        self.assertEqual(derived_status("NONFICTION", ""), "NEGATIVE")
        self.assertEqual(derived_status("FICTION", "UNCLEAR"), "UNRESOLVED")

    def test_secondary_grounding(self):
        from openlibrary.secondary_core import ground_evidence

        row = {"description": "A world changed by rising seas."}
        self.assertEqual(
            ground_evidence(row, "rising seas", "NO_EVIDENCE")["evidence_status"],
            "grounded",
        )
        self.assertEqual(
            ground_evidence(row, "climate migration", "NO_EVIDENCE")["evidence_status"],
            "invalid_not_found",
        )

    def test_final_annotations_and_routes(self):
        from collections import Counter
        from openlibrary.evaluation import derived_status, route_signature

        with (ROOT / "openlibrary/data/evaluation.csv").open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 100)
        self.assertEqual(len({r["work_key"] for r in rows}), 100)
        self.assertEqual(
            Counter(r["evaluation_stratum"] for r in rows),
            {
                s: 20
                for s in [
                    "explicit_only",
                    "motif_only",
                    "semantic_only",
                    "route_overlap",
                    "outside_pool",
                ]
            },
        )
        for row in rows:
            self.assertEqual(route_signature(row), row["evaluation_stratum"])
            self.assertEqual(
                derived_status(
                    row["fiction_status"], row["environment_climate_relevance"]
                ),
                row["derived_status"],
            )
        with (ROOT / "openlibrary/data/exclusions.csv").open() as f:
            excluded = {r["work_key"] for r in csv.DictReader(f)}
        self.assertTrue(excluded.isdisjoint(r["work_key"] for r in rows))

    def test_statistics(self):
        from openlibrary.evaluation import wilson_interval
        from openlibrary.statistics import grouped_binomial_trend
        import numpy as np

        low, high = wilson_interval(18, 20)
        self.assertAlmostEqual(low, 0.6989663548)
        self.assertAlmostEqual(high, 0.9721335188)
        result = grouped_binomial_trend(
            np.arange(2000, 2004), np.array([0, 0, 0, 0]), np.ones(4)
        )
        self.assertEqual(result["status"], "not_estimable")

    def test_secondary_reads_requested_data_root(self):
        from unittest.mock import patch
        import reproduce

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "openlibrary/data/secondary.csv"
            path.parent.mkdir(parents=True)
            with path.open("w") as f:
                f.write("gold_fiction_status\n")
            with patch.object(reproduce, "DATA", Path(d)):
                # An empty supplied dataset must not silently use packaged predictions.
                with self.assertRaisesRegex(ValueError, "requires predictions"):
                    reproduce.secondary(Path(d))

    def test_readme_and_public_scope(self):
        self.assertLessEqual(len((ROOT / "README.md").read_text().split()), 500)
        self.assertFalse((ROOT / "goodreads").exists())
        self.assertFalse((ROOT / "matching").exists())
        for line in (ROOT / "CHECKSUMS.sha256").read_text().splitlines():
            _, name = line.split("  ", 1)
            path = ROOT / name
            self.assertLess(path.stat().st_size, 5 * 1024 * 1024, name)

    def test_full_references(self):
        import full

        full.check(json.loads((ROOT / "pipeline.json").read_text()))

    def test_manifest(self):
        manifest = ROOT / "CHECKSUMS.sha256"
        self.assertTrue(manifest.is_file(), "Release manifest missing")
        for line in manifest.read_text().splitlines():
            expected, name = line.split("  ", 1)
            self.assertEqual(
                hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected, name
            )


@unittest.skipUnless(ARROW, "Install pyarrow to run the small pipeline checks")
class PipelineChecks(unittest.TestCase):
    def test_portable_modules(self):
        for path in sorted((ROOT / "openlibrary").glob("*.py")):
            importlib.import_module("openlibrary." + path.stem)

    def test_prefixes_and_truncation(self):
        from openlibrary.retrieval import document_text

        cfg = yaml.safe_load((ROOT / "openlibrary/config/evaluation.yaml").read_text())
        text = document_text(
            {
                "title": "Title",
                "subtitle": "Subtitle",
                "best_description": "D" * 6100,
                "subjects": [f"UNIQUE_SUBJECT_{i:02d}" for i in range(31)],
            },
            cfg,
        )
        self.assertTrue(text.startswith("passage: Title: Title Subtitle"))
        self.assertIn("D" * 6000, text)
        self.assertNotIn("D" * 6001, text)
        self.assertIn("UNIQUE_SUBJECT_29", text)
        self.assertNotIn("UNIQUE_SUBJECT_30", text)
        self.assertTrue(all(q["text"].startswith("query: ") for q in cfg["queries"]))
        self.assertEqual(cfg["model"]["max_sequence_length"], 512)

    def test_route_boundaries(self):
        from openlibrary.retrieval import lexical_features, stratum_for

        terms = yaml.safe_load((ROOT / "openlibrary/config/terms.yaml").read_text())

        def features(text):
            return lexical_features(text, terms)

        self.assertFalse(features("A flash flood and drought.")["motif_route"])
        self.assertTrue(features("A story of geoengineering.")["motif_route"])
        self.assertTrue(features("Rising seas and crop failures.")["motif_route"])
        self.assertFalse(features("Environmental fiction.")["explicit_route"])
        row = dict(semantic_rank=101, explicit_route=True, motif_route=True)
        self.assertEqual(stratum_for(row, 100), "")
        row["semantic_rank"] = 100
        self.assertEqual(stratum_for(row, 100), "route_overlap")

    def test_empty_temporal_period(self):
        from openlibrary.temporal import markdown_report

        text = markdown_report(
            {"claim_guardrail": ""},
            [
                {
                    "population": "all_works",
                    "field": "any",
                    "period": "2017-2025",
                    "works": 0,
                    "explicit_count": 0,
                    "rate_per_10000_works": None,
                }
            ],
            [{"role": "primary", "status": "not_estimable", "id": "fixture"}],
        )
        self.assertIn("0 | 0 | n/a", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
