"""Benchmark suite unit tests."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from gnn_common.metrics import compute_benchmark_metrics
from tools.benchmark import (
    load_benchmark_config,
    generate_run_matrix,
    aggregate_results,
    generate_comparison_report,
)
from tools.package_sample_dataset import parse_filename


class TestComputeBenchmarkMetrics(unittest.TestCase):
    """Test compute_benchmark_metrics()."""

    def test_basic_classification(self):
        """Core classification metrics are returned."""
        rng = np.random.RandomState(42)
        labels = rng.randint(0, 5, size=200)
        preds = labels.copy()
        # Introduce some errors
        preds[:20] = (preds[:20] + 1) % 5

        result = compute_benchmark_metrics(labels, preds, num_classes=5)

        self.assertIn("accuracy", result)
        self.assertIn("macro_f1", result)
        self.assertIn("weighted_f1", result)
        self.assertIn("balanced_accuracy", result)
        self.assertIn("mcc", result)
        self.assertGreater(result["accuracy"], 0.8)
        self.assertGreater(result["macro_f1"], 0.0)

    def test_with_probabilities(self):
        """Top-k accuracy and AUPRC are computed when probs provided."""
        rng = np.random.RandomState(42)
        n, c = 100, 5
        labels = rng.randint(0, c, size=n)
        probs = rng.dirichlet(np.ones(c), size=n)
        preds = probs.argmax(axis=1)

        result = compute_benchmark_metrics(labels, preds, num_classes=c, all_probs=probs)

        self.assertIn("top_1_accuracy", result)
        self.assertIn("top_3_accuracy", result)
        self.assertIn("auprc", result)

    def test_with_size_labels(self):
        """Size accuracy is computed for multi-task runs."""
        labels = np.array([0, 0, 1, 1, 2])
        preds = np.array([0, 0, 1, 2, 2])
        size_labels = np.array([0, 1, 2])
        size_preds = np.array([0, 1, 1])

        result = compute_benchmark_metrics(
            labels, preds, num_classes=3,
            size_labels=size_labels, size_preds=size_preds,
        )

        self.assertIn("size_accuracy", result)
        self.assertAlmostEqual(result["size_accuracy"], 2 / 3)

    def test_perfect_prediction(self):
        """Perfect predictions yield F1 = 1.0."""
        labels = np.array([0, 1, 2, 0, 1, 2])
        preds = labels.copy()

        result = compute_benchmark_metrics(labels, preds, num_classes=3)

        self.assertAlmostEqual(result["accuracy"], 1.0)
        self.assertAlmostEqual(result["macro_f1"], 1.0, places=5)


class TestBenchmarkConfig(unittest.TestCase):
    """Test config loading and run matrix generation."""

    def test_default_config(self):
        """Loading with no path returns defaults."""
        cfg = load_benchmark_config(None)
        self.assertIn("seeds", cfg)
        self.assertIn("split_types", cfg)
        self.assertIsInstance(cfg["seeds"], list)

    def test_yaml_config(self):
        """Loading from YAML file works."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("benchmark:\n  seeds: [1, 2]\n  split_types:\n    - type: iid\n")
            f.flush()
            cfg = load_benchmark_config(f.name)

        self.assertEqual(cfg["seeds"], [1, 2])
        Path(f.name).unlink()

    def test_missing_config_file(self):
        """Missing config file falls back to defaults."""
        cfg = load_benchmark_config("/nonexistent/path.yaml")
        self.assertIn("seeds", cfg)


class TestRunMatrix(unittest.TestCase):
    """Test run matrix generation."""

    def test_matrix_size(self):
        """Matrix has correct number of entries."""
        cfg = {
            "seeds": [42, 84],
            "split_types": [{"type": "iid"}, {"type": "layer", "params": {"train_layers": [1]}}],
            "models": [{"name": "GAT", "args": {}}],
            "output": {"prefix": "test"},
        }
        matrix = generate_run_matrix(cfg)
        self.assertEqual(len(matrix), 4)  # 2 seeds x 2 splits x 1 model

    def test_run_ids_unique(self):
        """All run IDs in the matrix are unique."""
        cfg = {
            "seeds": [42, 84, 126],
            "split_types": [{"type": "iid"}, {"type": "defect_size"}],
            "models": [{"name": "GAT", "args": {}}],
            "output": {"prefix": "bench"},
        }
        matrix = generate_run_matrix(cfg)
        run_ids = [m["run_id"] for m in matrix]
        self.assertEqual(len(run_ids), len(set(run_ids)))

    def test_matrix_contents(self):
        """Each run spec contains required keys."""
        cfg = {
            "seeds": [42],
            "split_types": [{"type": "iid"}],
            "models": [{"name": "M", "args": {"lr": 0.01}}],
            "output": {"prefix": "t"},
        }
        matrix = generate_run_matrix(cfg)
        spec = matrix[0]
        self.assertIn("run_id", spec)
        self.assertIn("seed", spec)
        self.assertIn("split_type", spec)
        self.assertIn("model_args", spec)
        self.assertEqual(spec["seed"], 42)
        self.assertEqual(spec["split_type"], "iid")


class TestAggregateResults(unittest.TestCase):
    """Test result aggregation."""

    def test_aggregate_empty(self):
        """Empty input produces empty DataFrame."""
        df = aggregate_results([])
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 0)

    def test_aggregate_basic(self):
        """Basic aggregation produces correct columns."""
        results = [
            {
                "run_info": {"run_id": "r1", "split_type": "iid", "seed": 42, "model_name": "GAT"},
                "metrics": {
                    "summary": {"exit_code": 0},
                    "metrics": {
                        "test": {"accuracy": 0.95, "macro_f1": 0.8, "weighted_f1": 0.9,
                                 "balanced_accuracy": 0.85, "mcc": 0.7},
                        "best": {"epoch": 100, "macro_f1": 0.82},
                    },
                },
            }
        ]
        df = aggregate_results(results)
        self.assertEqual(len(df), 1)
        self.assertAlmostEqual(df.iloc[0]["test_macro_f1"], 0.8)


class TestReportGeneration(unittest.TestCase):
    """Test report output."""

    def test_generates_files(self):
        """Report generates CSV, JSON, and summary files."""
        df = pd.DataFrame({
            "run_id": ["r1"],
            "split_type": ["iid"],
            "test_macro_f1": [0.8],
            "test_weighted_f1": [0.9],
            "test_balanced_accuracy": [0.85],
            "test_mcc": [0.7],
        })
        with tempfile.TemporaryDirectory() as tmp:
            generate_comparison_report(df, Path(tmp))
            self.assertTrue((Path(tmp) / "benchmark_results.csv").exists())
            self.assertTrue((Path(tmp) / "benchmark_results.json").exists())
            self.assertTrue((Path(tmp) / "benchmark_summary.txt").exists())


class TestParseFilename(unittest.TestCase):
    """Test filename parsing for sample dataset packager."""

    def test_defect_file(self):
        meta = parse_filename("Defect_L10_B100_el1165_H4_W4.npy")
        self.assertEqual(meta["type"], "defect")
        self.assertEqual(meta["layer"], 10)
        self.assertEqual(meta["h"], 4)
        self.assertEqual(meta["size_class"], "medium")

    def test_small_defect(self):
        meta = parse_filename("Defect_L1_B1_el100_H2_W2.npy")
        self.assertEqual(meta["size_class"], "small")

    def test_large_defect(self):
        meta = parse_filename("Defect_L2_B50_el500_H8_W8.npy")
        self.assertEqual(meta["size_class"], "large")

    def test_noise_defect_free(self):
        meta = parse_filename("NoiseDefectFree_something.npy")
        self.assertEqual(meta["type"], "noise_defect_free")


if __name__ == "__main__":
    unittest.main()
