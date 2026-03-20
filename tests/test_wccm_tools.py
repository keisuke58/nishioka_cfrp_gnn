"""Unit tests for WCCM tool scripts.

Tests cover:
  - tools/run_ir_comparison.py  (argument parsing, save_results_csv, split_data stub)
  - tools/wccm_visualize.py     (class labels, visualization modes, LaTeX table)
  - tools/prepare_1x1_ood.py    (size extraction, config YAML, file lists)
"""

import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import yaml

# Ensure project root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ============================================================================
# Tests for run_ir_comparison.py
# ============================================================================

class TestIRComparisonArgParsing(unittest.TestCase):
    """Test argument parsing for run_ir_comparison.py."""

    def _parse(self, args_list):
        """Helper: parse args without hitting sys.argv."""
        from tools.run_ir_comparison import parse_args
        with patch("sys.argv", ["run_ir_comparison.py"] + args_list):
            return parse_args()

    def test_defaults(self):
        """Default argument values are correct."""
        args = self._parse([])
        self.assertEqual(args.seeds, [42, 123, 456])
        self.assertEqual(args.epochs, 150)
        self.assertEqual(args.split, "iid")
        self.assertEqual(args.patience, 50)
        self.assertEqual(args.batch_size, 64)
        self.assertEqual(args.hidden_channels, 64)
        self.assertAlmostEqual(args.lr, 0.002)
        self.assertIsNone(args.output_csv)
        self.assertAlmostEqual(args.data_usage_ratio, 1.0)

    def test_custom_seeds(self):
        """Custom seeds are parsed correctly."""
        args = self._parse(["--seeds", "1", "2", "3"])
        self.assertEqual(args.seeds, [1, 2, 3])

    def test_custom_epochs_and_lr(self):
        """Custom epochs and lr are parsed."""
        args = self._parse(["--epochs", "50", "--lr", "0.001"])
        self.assertEqual(args.epochs, 50)
        self.assertAlmostEqual(args.lr, 0.001)

    def test_data_usage_ratio(self):
        """data_usage_ratio flag is parsed."""
        args = self._parse(["--data_usage_ratio", "0.3"])
        self.assertAlmostEqual(args.data_usage_ratio, 0.3)

    def test_output_csv(self):
        """output_csv can be set."""
        args = self._parse(["--output_csv", "/tmp/results.csv"])
        self.assertEqual(args.output_csv, "/tmp/results.csv")


class TestIRComparisonImport(unittest.TestCase):
    """Test that the script module can be imported without side effects."""

    def test_import_module(self):
        """Importing the module does not raise."""
        import tools.run_ir_comparison as mod  # noqa: F401
        self.assertTrue(hasattr(mod, "parse_args"))
        self.assertTrue(hasattr(mod, "run_comparison"))
        self.assertTrue(hasattr(mod, "save_results_csv"))
        self.assertTrue(hasattr(mod, "build_loss_fn"))
        self.assertTrue(hasattr(mod, "evaluate"))


class TestIRComparisonSaveResults(unittest.TestCase):
    """Test save_results_csv writes correct CSV."""

    def test_save_results_csv(self):
        """CSV is written with correct header and rows."""
        from tools.run_ir_comparison import save_results_csv

        results = [
            {"seed": 42, "config": "FEM_4ch", "macro_f1": 0.85, "accuracy": 0.90},
            {"seed": 42, "config": "FEM_IR_7ch", "macro_f1": 0.88, "accuracy": 0.92},
        ]

        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "sub", "results.csv")
            save_results_csv(results, out_path)

            self.assertTrue(os.path.exists(out_path))

            with open(out_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["seed"], "42")
            self.assertEqual(rows[0]["config"], "FEM_4ch")
            self.assertIn("macro_f1", rows[0])

    def test_save_empty_results(self):
        """Empty results list produces no file."""
        from tools.run_ir_comparison import save_results_csv

        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "empty.csv")
            save_results_csv([], out_path)
            self.assertFalse(os.path.exists(out_path))


class TestIRComparisonPrintSummary(unittest.TestCase):
    """Test print_summary_table does not crash."""

    def test_print_summary_table(self):
        """print_summary_table runs without error on valid data."""
        from tools.run_ir_comparison import print_summary_table

        results = [
            {"seed": 42, "config": "FEM_4ch", "input_channels": 4,
             "macro_f1": 0.80, "mcc": 0.70, "accuracy": 0.85,
             "weighted_f1": 0.82, "balanced_accuracy": 0.78,
             "best_epoch": 100, "train_time_sec": 60.0},
            {"seed": 42, "config": "FEM_IR_7ch", "input_channels": 7,
             "macro_f1": 0.85, "mcc": 0.75, "accuracy": 0.88,
             "weighted_f1": 0.87, "balanced_accuracy": 0.83,
             "best_epoch": 90, "train_time_sec": 80.0},
        ]

        # Should not raise
        print_summary_table(results)


# ============================================================================
# Tests for wccm_visualize.py
# ============================================================================

class TestWccmVisualizeClassLabels(unittest.TestCase):
    """Test class label generation."""

    def test_english_labels_count(self):
        """There are exactly 19 English class labels."""
        from tools.wccm_visualize import CLASS_LABELS_EN
        self.assertEqual(len(CLASS_LABELS_EN), 19)

    def test_japanese_labels_count(self):
        """There are exactly 19 Japanese class labels."""
        from tools.wccm_visualize import CLASS_LABELS_JA
        self.assertEqual(len(CLASS_LABELS_JA), 19)

    def test_short_labels_count(self):
        """SHORT_LABELS has 19 entries starting with DF."""
        from tools.wccm_visualize import SHORT_LABELS
        self.assertEqual(len(SHORT_LABELS), 19)
        self.assertEqual(SHORT_LABELS[0], "DF")
        self.assertEqual(SHORT_LABELS[1], "C1")
        self.assertEqual(SHORT_LABELS[18], "C18")

    def test_get_labels_en(self):
        """_get_labels('en') returns English labels."""
        from tools.wccm_visualize import _get_labels, CLASS_LABELS_EN
        self.assertEqual(_get_labels("en"), CLASS_LABELS_EN)

    def test_get_labels_ja(self):
        """_get_labels('ja') returns Japanese labels."""
        from tools.wccm_visualize import _get_labels, CLASS_LABELS_JA
        self.assertEqual(_get_labels("ja"), CLASS_LABELS_JA)

    def test_defect_free_label(self):
        """Class 0 is 'Defect-free' in English."""
        from tools.wccm_visualize import CLASS_LABELS_EN
        self.assertEqual(CLASS_LABELS_EN[0], "Defect-free")


class TestWccmVisualizeSplitNames(unittest.TestCase):
    """Test split display names."""

    def test_split_names_en(self):
        from tools.wccm_visualize import _get_split_names
        names = _get_split_names("en")
        self.assertIn("iid", names)
        self.assertEqual(names["iid"], "IID")

    def test_split_names_ja(self):
        from tools.wccm_visualize import _get_split_names
        names = _get_split_names("ja")
        self.assertIn("iid", names)
        self.assertEqual(names["iid"], "IID")


class TestWccmVisualizeModes(unittest.TestCase):
    """Test that visualization mode functions exist and are callable."""

    def test_mode_funcs_dict(self):
        """MODE_FUNCS has all expected keys."""
        from tools.wccm_visualize import MODE_FUNCS
        expected = {"confusion", "f1_comparison", "ood_comparison",
                    "training_curve", "summary_table"}
        self.assertEqual(set(MODE_FUNCS.keys()), expected)

    def test_mode_funcs_are_callable(self):
        """All mode functions are callable."""
        from tools.wccm_visualize import MODE_FUNCS
        for name, func in MODE_FUNCS.items():
            self.assertTrue(callable(func), f"Mode '{name}' is not callable")


class TestWccmVisualizeConfusionMatrix(unittest.TestCase):
    """Test confusion matrix plotting."""

    def test_plot_confusion_matrix_normalized(self):
        """plot_confusion_matrix creates a file (normalized)."""
        from tools.wccm_visualize import plot_confusion_matrix

        cm = np.eye(5, dtype=int) * 10
        cm[0, 1] = 2
        labels = [f"C{i}" for i in range(5)]

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "cm.pdf"
            plot_confusion_matrix(cm, labels, out_path, normalize=True)
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)

    def test_plot_confusion_matrix_raw(self):
        """plot_confusion_matrix creates a file (raw counts)."""
        from tools.wccm_visualize import plot_confusion_matrix

        cm = np.eye(3, dtype=int) * 20
        labels = ["A", "B", "C"]

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "cm_raw.pdf"
            plot_confusion_matrix(cm, labels, out_path, normalize=False)
            self.assertTrue(out_path.exists())

    def test_plot_confusion_matrix_19x19(self):
        """plot_confusion_matrix works with 19 classes."""
        from tools.wccm_visualize import plot_confusion_matrix, CLASS_LABELS_EN

        rng = np.random.RandomState(42)
        cm = np.zeros((19, 19), dtype=int)
        for i in range(19):
            cm[i, i] = rng.randint(50, 200)
            for j in range(19):
                if j != i:
                    cm[i, j] = rng.randint(0, 5)

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "cm19.pdf"
            plot_confusion_matrix(cm, CLASS_LABELS_EN, out_path)
            self.assertTrue(out_path.exists())


class TestWccmVisualizeExtractHelpers(unittest.TestCase):
    """Test data extraction helper functions."""

    def test_extract_confusion_matrix_direct(self):
        """_extract_confusion_matrix with direct key."""
        from tools.wccm_visualize import _extract_confusion_matrix
        cm_list = [[10, 1], [2, 8]]
        result = _extract_confusion_matrix({"confusion_matrix": cm_list})
        self.assertIsNotNone(result)
        np.testing.assert_array_equal(result, np.array(cm_list))

    def test_extract_confusion_matrix_nested(self):
        """_extract_confusion_matrix from test sub-dict."""
        from tools.wccm_visualize import _extract_confusion_matrix
        cm_list = [[5, 0], [1, 4]]
        result = _extract_confusion_matrix({"test": {"cm": cm_list}})
        self.assertIsNotNone(result)
        np.testing.assert_array_equal(result, np.array(cm_list))

    def test_extract_confusion_matrix_missing(self):
        """_extract_confusion_matrix returns None when absent."""
        from tools.wccm_visualize import _extract_confusion_matrix
        result = _extract_confusion_matrix({"accuracy": 0.9})
        self.assertIsNone(result)

    def test_extract_per_class_f1_direct(self):
        """_extract_per_class_f1 with direct key."""
        from tools.wccm_visualize import _extract_per_class_f1
        f1 = [0.9, 0.8, 0.7]
        result = _extract_per_class_f1({"f1_per_class": f1})
        np.testing.assert_array_almost_equal(result, np.array(f1))

    def test_extract_per_class_f1_nested(self):
        """_extract_per_class_f1 from test sub-dict."""
        from tools.wccm_visualize import _extract_per_class_f1
        f1 = [0.5, 0.6]
        result = _extract_per_class_f1({"test": {"per_class_f1": f1}})
        np.testing.assert_array_almost_equal(result, np.array(f1))

    def test_extract_per_class_f1_missing(self):
        """_extract_per_class_f1 returns None when absent."""
        from tools.wccm_visualize import _extract_per_class_f1
        result = _extract_per_class_f1({"mcc": 0.5})
        self.assertIsNone(result)


class TestWccmVisualizeSummaryTable(unittest.TestCase):
    """Test LaTeX summary table generation."""

    def test_mode_summary_table_from_csv(self):
        """mode_summary_table generates .tex and .csv from benchmark CSV input."""
        from tools.wccm_visualize import mode_summary_table

        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp) / "results"
            results_dir.mkdir()
            output_dir = Path(tmp) / "output"

            # Create a benchmark_results.csv
            df = pd.DataFrame({
                "run_id": ["run_iid_42", "run_iid_84", "run_defect_size_42"],
                "split_type": ["iid", "iid", "defect_size"],
                "test_accuracy": [0.90, 0.88, 0.75],
                "test_macro_f1": [0.80, 0.78, 0.60],
                "test_mcc": [0.70, 0.68, 0.50],
            })
            df.to_csv(results_dir / "benchmark_results.csv", index=False)

            mode_summary_table(results_dir, output_dir, lang="en")

            self.assertTrue((output_dir / "summary_table.tex").exists())
            self.assertTrue((output_dir / "summary_table.csv").exists())

            # Verify LaTeX content
            tex_content = (output_dir / "summary_table.tex").read_text()
            self.assertIn(r"\begin{table}", tex_content)
            self.assertIn(r"\toprule", tex_content)
            self.assertIn("IID", tex_content)

    def test_mode_summary_table_from_metrics_json(self):
        """mode_summary_table generates table from metrics.json files."""
        from tools.wccm_visualize import mode_summary_table

        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp) / "results"
            output_dir = Path(tmp) / "output"

            # Create a run sub-directory with metrics.json
            run_dir = results_dir / "run_iid_seed42"
            run_dir.mkdir(parents=True)
            metrics = {
                "run_id": "run_iid_seed42",
                "test": {
                    "accuracy": 0.92,
                    "macro_f1": 0.85,
                    "weighted_f1": 0.88,
                    "balanced_accuracy": 0.80,
                    "mcc": 0.72,
                },
            }
            with open(run_dir / "metrics.json", "w") as f:
                json.dump(metrics, f)

            mode_summary_table(results_dir, output_dir, lang="en")
            self.assertTrue((output_dir / "summary_table.tex").exists())


class TestWccmVisualizeTrainingCurve(unittest.TestCase):
    """Test training curve plotting from legacy CSV."""

    def test_training_curve_legacy_csv(self):
        """mode_training_curve generates PDF from legacy loss CSV."""
        from tools.wccm_visualize import mode_training_curve

        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp) / "results"
            results_dir.mkdir()
            output_dir = Path(tmp) / "output"

            # Create a legacy loss CSV
            epochs = list(range(50))
            df = pd.DataFrame({
                "Epoch": epochs,
                "Training Loss": [1.0 / (e + 1) for e in epochs],
                "Validation Loss": [1.2 / (e + 1) for e in epochs],
            })
            df.to_csv(results_dir / "loss_data_0.0079_20240927.csv", index=False)

            mode_training_curve(results_dir, output_dir, csv_pattern="loss_data_*.csv")
            self.assertTrue((output_dir / "training_curves_legacy.pdf").exists())


class TestWccmVisualizeOodComparison(unittest.TestCase):
    """Test OOD comparison bar chart."""

    def test_ood_comparison_from_csv(self):
        """mode_ood_comparison generates PDF from benchmark CSV."""
        from tools.wccm_visualize import mode_ood_comparison

        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp) / "results"
            results_dir.mkdir()
            output_dir = Path(tmp) / "output"

            df = pd.DataFrame({
                "run_id": ["r1", "r2", "r3"],
                "split_type": ["iid", "iid", "defect_size"],
                "test_macro_f1": [0.80, 0.82, 0.60],
                "test_mcc": [0.70, 0.72, 0.50],
                "test_accuracy": [0.90, 0.91, 0.75],
            })
            df.to_csv(results_dir / "benchmark_results.csv", index=False)

            mode_ood_comparison(results_dir, output_dir, lang="en")
            self.assertTrue((output_dir / "ood_comparison.pdf").exists())


class TestWccmVisualizeF1Comparison(unittest.TestCase):
    """Test F1 comparison bar chart."""

    def test_f1_comparison_from_metrics_json(self):
        """mode_f1_comparison generates PDF from per-class F1 in metrics.json."""
        from tools.wccm_visualize import mode_f1_comparison

        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp) / "results"
            output_dir = Path(tmp) / "output"

            run_dir = results_dir / "run_iid_42"
            run_dir.mkdir(parents=True)
            metrics = {
                "run_id": "run_iid_42",
                "f1_per_class": [0.9 - i * 0.02 for i in range(19)],
            }
            with open(run_dir / "metrics.json", "w") as f:
                json.dump(metrics, f)

            mode_f1_comparison(results_dir, output_dir, lang="en")
            self.assertTrue((output_dir / "f1_per_class_comparison.pdf").exists())


# ============================================================================
# Tests for prepare_1x1_ood.py
# ============================================================================

class TestPrepare1x1OodSizeExtraction(unittest.TestCase):
    """Test size class detection from filenames."""

    def test_extract_size_8x8(self):
        """Extract (8, 8) from H8_W8 filename."""
        from tools.prepare_1x1_ood import extract_size
        result = extract_size("Defect_L2_B3_el129_H8_W8.npy")
        self.assertEqual(result, ("8", "8"))

    def test_extract_size_4x4(self):
        """Extract (4, 4) from H4_W4 filename."""
        from tools.prepare_1x1_ood import extract_size
        result = extract_size("Defect_L10_B100_el1165_H4_W4.npy")
        self.assertEqual(result, ("4", "4"))

    def test_extract_size_2x2(self):
        """Extract (2, 2) from H2_W2 filename."""
        from tools.prepare_1x1_ood import extract_size
        result = extract_size("Defect_L5_B50_el500_H2_W2.npy")
        self.assertEqual(result, ("2", "2"))

    def test_extract_size_1x1(self):
        """Extract (1, 1) from H1_W1 filename."""
        from tools.prepare_1x1_ood import extract_size
        result = extract_size("Defect_L3_B10_el200_H1_W1.npy")
        self.assertEqual(result, ("1", "1"))

    def test_extract_size_no_match(self):
        """Returns None when no H_W pattern is present."""
        from tools.prepare_1x1_ood import extract_size
        result = extract_size("NoiseDefectFree_something.npy")
        self.assertIsNone(result)

    def test_extract_size_label_file(self):
        """Extracts size even from label filenames."""
        from tools.prepare_1x1_ood import extract_size
        result = extract_size("Defect_L2_B3_el129_H8_W8_19label.npy")
        self.assertEqual(result, ("8", "8"))


class TestPrepare1x1OodLabelFile(unittest.TestCase):
    """Test label filename construction."""

    def test_label_file_for_data(self):
        """label_file_for_data appends _19label."""
        from tools.prepare_1x1_ood import label_file_for_data
        result = label_file_for_data("Defect_L2_B3_el129_H8_W8.npy")
        self.assertEqual(result, "Defect_L2_B3_el129_H8_W8_19label.npy")

    def test_label_file_for_noise(self):
        """label_file_for_data works for noise files."""
        from tools.prepare_1x1_ood import label_file_for_data
        result = label_file_for_data("NoiseDefectFree_abc.npy")
        self.assertEqual(result, "NoiseDefectFree_abc_19label.npy")


class TestPrepare1x1OodConfigGeneration(unittest.TestCase):
    """Test YAML config generation."""

    def test_create_ood_config(self):
        """create_ood_config writes valid YAML with expected structure."""
        from tools.prepare_1x1_ood import create_ood_config

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "sub" / "config.yaml"
            create_ood_config(config_path)

            self.assertTrue(config_path.exists())

            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            self.assertIn("benchmark", config)
            bench = config["benchmark"]
            self.assertEqual(bench["name"], "cfrp_gnn_1x1_micro_defect_ood")
            self.assertIn("seeds", bench)
            self.assertEqual(bench["seeds"], [42, 84, 126])
            self.assertIn("split_types", bench)
            self.assertEqual(len(bench["split_types"]), 1)
            self.assertEqual(bench["split_types"][0]["type"], "property_ood")
            self.assertIn("data", bench)
            self.assertEqual(bench["data"]["num_classes"], 19)

    def test_create_ood_config_creates_parent_dirs(self):
        """create_ood_config creates parent directories."""
        from tools.prepare_1x1_ood import create_ood_config

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "a" / "b" / "c" / "config.yaml"
            create_ood_config(config_path)
            self.assertTrue(config_path.exists())


class TestPrepare1x1OodFileListCreation(unittest.TestCase):
    """Test file list creation with mock directory structure."""

    def _setup_mock_data(self, tmp_path: Path):
        """Create a mock data directory with various size files."""
        data_dir = tmp_path / "data"
        label_dir = tmp_path / "labels"

        # Create train/val/test subdirectories
        for split in ("train", "val", "test"):
            (data_dir / split).mkdir(parents=True)

        # Create H8_W8 files in train
        for i in range(3):
            name = f"Defect_L2_B{i}_el{i*100}_H8_W8.npy"
            np.save(str(data_dir / "train" / name), np.zeros(10))
            # Create corresponding label
            label_dir.mkdir(parents=True, exist_ok=True)
            label_name = f"Defect_L2_B{i}_el{i*100}_H8_W8_19label.npy"
            np.save(str(label_dir / label_name), np.zeros(10))

        # Create H4_W4 files
        for i in range(2):
            name = f"Defect_L3_B{i}_el{i*50}_H4_W4.npy"
            np.save(str(data_dir / "train" / name), np.zeros(10))
            label_name = f"Defect_L3_B{i}_el{i*50}_H4_W4_19label.npy"
            np.save(str(label_dir / label_name), np.zeros(10))

        # Create H1_W1 files in test
        for i in range(2):
            name = f"Defect_L5_B{i}_el{i*20}_H1_W1.npy"
            np.save(str(data_dir / "test" / name), np.zeros(10))
            label_name = f"Defect_L5_B{i}_el{i*20}_H1_W1_19label.npy"
            np.save(str(label_dir / label_name), np.zeros(10))

        return data_dir, label_dir

    def test_find_npy_files(self):
        """find_npy_files discovers files across subdirectories."""
        from tools.prepare_1x1_ood import find_npy_files

        with tempfile.TemporaryDirectory() as tmp:
            data_dir, _ = self._setup_mock_data(Path(tmp))
            files = find_npy_files(data_dir)
            self.assertEqual(len(files), 7)  # 3 + 2 + 2

    def test_create_file_lists(self):
        """create_file_lists writes train and test file lists."""
        from tools.prepare_1x1_ood import create_file_lists

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir, label_dir = self._setup_mock_data(tmp_path)
            output_dir = tmp_path / "output"

            # Monkey-patch REPO_ROOT for the relative path computation
            with patch("tools.prepare_1x1_ood.REPO_ROOT", tmp_path):
                train_files, test_files, lm_train, lm_test = create_file_lists(
                    data_dir, label_dir, output_dir,
                )

            # H8_W8 (3) + H4_W4 (2) = 5 train files
            self.assertEqual(len(train_files), 5)
            # H1_W1 (2) = 2 test files
            self.assertEqual(len(test_files), 2)
            # All labels exist
            self.assertEqual(lm_train, 0)
            self.assertEqual(lm_test, 0)

            # Check file lists were written
            self.assertTrue((output_dir / "train_files.txt").exists())
            self.assertTrue((output_dir / "test_files.txt").exists())

            # Check content
            train_content = (output_dir / "train_files.txt").read_text()
            self.assertIn("H8_W8", train_content)
            self.assertIn("H4_W4", train_content)

            test_content = (output_dir / "test_files.txt").read_text()
            self.assertIn("H1_W1", test_content)

    def test_create_file_lists_missing_labels(self):
        """create_file_lists reports missing labels."""
        from tools.prepare_1x1_ood import create_file_lists

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "data"
            label_dir = tmp_path / "labels_empty"
            label_dir.mkdir(parents=True)
            output_dir = tmp_path / "output"

            # Create data files without labels
            (data_dir / "train").mkdir(parents=True)
            np.save(str(data_dir / "train" / "Defect_L2_B0_el0_H8_W8.npy"), np.zeros(5))

            with patch("tools.prepare_1x1_ood.REPO_ROOT", tmp_path):
                train_files, test_files, lm_train, lm_test = create_file_lists(
                    data_dir, label_dir, output_dir,
                )

            self.assertEqual(len(train_files), 1)
            self.assertEqual(lm_train, 1)  # label missing


class TestPrepare1x1OodUpdateMainConfig(unittest.TestCase):
    """Test updating the main benchmark config YAML."""

    def test_update_main_config_adds_entry(self):
        """update_main_config adds property_ood_1x1 entry."""
        from tools.prepare_1x1_ood import update_main_config

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "benchmark_config.yaml"
            initial = {
                "benchmark": {
                    "seeds": [42],
                    "split_types": [{"type": "iid"}],
                }
            }
            with open(config_path, "w") as f:
                yaml.dump(initial, f)

            update_main_config(config_path)

            with open(config_path, "r") as f:
                updated = yaml.safe_load(f)

            splits = updated["benchmark"]["split_types"]
            self.assertEqual(len(splits), 2)
            names = [s.get("name") or s.get("type") for s in splits]
            self.assertIn("property_ood_1x1", names)

    def test_update_main_config_idempotent(self):
        """update_main_config does not duplicate if already present."""
        from tools.prepare_1x1_ood import update_main_config

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "benchmark_config.yaml"
            initial = {
                "benchmark": {
                    "seeds": [42],
                    "split_types": [
                        {"type": "iid"},
                        {"type": "property_ood_1x1"},
                    ],
                }
            }
            with open(config_path, "w") as f:
                yaml.dump(initial, f)

            update_main_config(config_path)

            with open(config_path, "r") as f:
                updated = yaml.safe_load(f)

            splits = updated["benchmark"]["split_types"]
            self.assertEqual(len(splits), 2)

    def test_update_main_config_missing_file(self):
        """update_main_config does not crash when file is missing."""
        from tools.prepare_1x1_ood import update_main_config

        # Should not raise
        update_main_config(Path("/nonexistent/config.yaml"))


class TestPrepare1x1OodValidateLabels(unittest.TestCase):
    """Test label validation reporting."""

    def test_validate_labels_all_present(self):
        """validate_labels returns True when no labels are missing."""
        from tools.prepare_1x1_ood import validate_labels

        with tempfile.TemporaryDirectory() as tmp:
            label_dir = Path(tmp) / "labels"
            label_dir.mkdir()
            result = validate_labels(
                train_files=["a.npy", "b.npy"],
                test_files=["c.npy"],
                label_dir=label_dir,
                label_missing_train=0,
                label_missing_test=0,
            )
            self.assertTrue(result)

    def test_validate_labels_some_missing(self):
        """validate_labels returns False when labels are missing."""
        from tools.prepare_1x1_ood import validate_labels

        with tempfile.TemporaryDirectory() as tmp:
            label_dir = Path(tmp) / "labels"
            label_dir.mkdir()
            result = validate_labels(
                train_files=["a.npy"],
                test_files=["b.npy"],
                label_dir=label_dir,
                label_missing_train=1,
                label_missing_test=0,
            )
            self.assertFalse(result)


class TestPrepare1x1OodConstants(unittest.TestCase):
    """Test module-level constants."""

    def test_train_sizes(self):
        """TRAIN_SIZES contains expected size tuples."""
        from tools.prepare_1x1_ood import TRAIN_SIZES
        self.assertIn(("2", "2"), TRAIN_SIZES)
        self.assertIn(("4", "4"), TRAIN_SIZES)
        self.assertIn(("8", "8"), TRAIN_SIZES)
        self.assertNotIn(("1", "1"), TRAIN_SIZES)

    def test_test_size(self):
        """TEST_SIZE contains only (1, 1)."""
        from tools.prepare_1x1_ood import TEST_SIZE
        self.assertEqual(TEST_SIZE, {("1", "1")})


if __name__ == "__main__":
    unittest.main()
