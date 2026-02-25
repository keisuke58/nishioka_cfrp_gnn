"""HPO and fine-tuning comparison unit tests."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import optuna
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.optuna_hpo import (
    load_hpo_config,
    sample_hyperparams,
    sample_finetune_params,
    build_training_args,
    create_study,
    export_results,
)
from tools.finetune_comparison import load_finetune_config, DEFAULT_STRATEGIES
from tools.analyze_sweeps import load_optuna_study


class TestHPOConfig(unittest.TestCase):
    """Test HPO config loading."""

    def test_default_config(self):
        cfg = load_hpo_config(None)
        self.assertIn("study_name", cfg)
        self.assertIn("search_space", cfg)
        self.assertIn("fixed_params", cfg)
        self.assertIsInstance(cfg["search_space"], dict)

    def test_yaml_config(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("hpo:\n  study_name: test_study\n  n_trials: 5\n")
            f.flush()
            cfg = load_hpo_config(f.name)

        self.assertEqual(cfg["study_name"], "test_study")
        self.assertEqual(cfg["n_trials"], 5)
        # defaults should be filled in
        self.assertIn("search_space", cfg)
        Path(f.name).unlink()

    def test_missing_config(self):
        cfg = load_hpo_config("/nonexistent.yaml")
        self.assertIn("study_name", cfg)


class TestSearchSpace(unittest.TestCase):
    """Test hyperparameter sampling."""

    def test_sample_hyperparams(self):
        study = optuna.create_study(direction="maximize")
        trial = study.ask()

        search_space = {
            "lr": {"type": "log_float", "low": 1e-4, "high": 1e-2},
            "batch": {"type": "categorical", "choices": [32, 64]},
            "drop": {"type": "float", "low": 0.0, "high": 0.5},
        }
        params = sample_hyperparams(trial, search_space)

        self.assertIn("lr", params)
        self.assertIn("batch", params)
        self.assertIn("drop", params)
        self.assertGreater(params["lr"], 0)
        self.assertIn(params["batch"], [32, 64])
        self.assertGreaterEqual(params["drop"], 0.0)
        self.assertLessEqual(params["drop"], 0.5)

    def test_sample_finetune_params(self):
        study = optuna.create_study(direction="maximize")
        trial = study.ask()

        params = sample_finetune_params(trial)
        self.assertIn("epochs", params)
        self.assertIn("patience", params)
        # Should have at least one strategy-specific param
        has_strategy_param = (
            "freeze_backbone" in params
            or "backbone_lr" in params
            or "learning_rate" in params
        )
        self.assertTrue(has_strategy_param)


class TestBuildTrainingArgs(unittest.TestCase):
    """Test training args construction."""

    def test_basic_args(self):
        params = {"learning_rate": 0.001, "batch_size": 64}
        fixed = {"epochs": 100, "use_amp": True}
        args = build_training_args(params, fixed)

        self.assertIn("--", args)
        self.assertIn("--learning_rate", args)
        self.assertIn("0.001", args)
        self.assertIn("--use_amp", args)

    def test_bool_false_excluded(self):
        params = {"use_logit_adjust": False}
        args = build_training_args(params, {})
        self.assertNotIn("--use_logit_adjust", args)

    def test_bool_true_included(self):
        params = {"use_amp": True}
        args = build_training_args(params, {})
        self.assertIn("--use_amp", args)


class TestStudyCreation(unittest.TestCase):
    """Test Optuna study creation."""

    def test_create_study_sqlite(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "study_name": "test_create",
                "direction": "maximize",
                "pruner": {"type": "median", "n_startup_trials": 3, "n_warmup_steps": 10},
                "sampler": {"type": "tpe", "seed": 42},
                "output": {"study_db_dir": tmp},
            }
            study = create_study(config)
            self.assertEqual(study.study_name, "test_create")
            self.assertTrue((Path(tmp) / "test_create.db").exists())

    def test_resume_study(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "study_name": "test_resume",
                "direction": "maximize",
                "pruner": {"type": "nop"},
                "sampler": {"type": "random", "seed": 0},
                "output": {"study_db_dir": tmp},
            }
            # Create and add a trial
            study1 = create_study(config)
            study1.add_trial(
                optuna.trial.create_trial(
                    params={"lr": 0.001},
                    distributions={"lr": optuna.distributions.FloatDistribution(1e-4, 1e-2)},
                    values=[0.7],
                )
            )
            self.assertEqual(len(study1.trials), 1)

            # Resume
            study2 = create_study(config)
            self.assertEqual(len(study2.trials), 1)


class TestExportResults(unittest.TestCase):
    """Test result export."""

    def test_export_with_trials(self):
        study = optuna.create_study(study_name="test_export", direction="maximize")
        for i in range(5):
            study.add_trial(
                optuna.trial.create_trial(
                    params={"lr": 0.001 * (i + 1), "batch": 64},
                    distributions={
                        "lr": optuna.distributions.FloatDistribution(1e-4, 1e-2),
                        "batch": optuna.distributions.CategoricalDistribution([32, 64]),
                    },
                    values=[0.6 + i * 0.02],
                )
            )

        with tempfile.TemporaryDirectory() as tmp:
            export_results(study, Path(tmp))
            self.assertTrue((Path(tmp) / "hpo_trials.csv").exists())
            self.assertTrue((Path(tmp) / "hpo_summary.json").exists())
            self.assertTrue((Path(tmp) / "hpo_summary.txt").exists())

            with open(Path(tmp) / "hpo_summary.json") as f:
                summary = json.load(f)
            self.assertEqual(summary["n_trials"], 5)
            self.assertIsNotNone(summary["best_trial"])

    def test_export_empty_study(self):
        study = optuna.create_study(study_name="test_empty", direction="maximize")
        with tempfile.TemporaryDirectory() as tmp:
            export_results(study, Path(tmp))
            self.assertTrue((Path(tmp) / "hpo_summary.json").exists())


class TestFinetuneConfig(unittest.TestCase):
    """Test fine-tuning comparison config."""

    def test_default_strategies(self):
        cfg = load_finetune_config(None)
        strategies = cfg["strategies"]
        self.assertEqual(len(strategies), 3)
        names = [s["name"] for s in strategies]
        self.assertIn("freeze_backbone", names)
        self.assertIn("differential_lr", names)
        self.assertIn("higher_lr", names)

    def test_default_seeds(self):
        cfg = load_finetune_config(None)
        self.assertEqual(cfg["seeds"], [42, 84, 126])


class TestOptunaStudyLoader(unittest.TestCase):
    """Test loading Optuna study for analyze_sweeps integration."""

    def test_load_study_db(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            storage = f"sqlite:///{db_path}"

            study = optuna.create_study(
                study_name="loader_test",
                storage=storage,
                direction="maximize",
            )
            study.add_trial(
                optuna.trial.create_trial(
                    params={"lr": 0.002, "batch": 64},
                    distributions={
                        "lr": optuna.distributions.FloatDistribution(1e-4, 1e-2),
                        "batch": optuna.distributions.CategoricalDistribution([32, 64]),
                    },
                    values=[0.73],
                )
            )

            df = load_optuna_study(str(db_path))
            self.assertGreater(len(df), 0)
            self.assertIn("best_macro_f1", df.columns)
            self.assertAlmostEqual(df.iloc[0]["best_macro_f1"], 0.73)


if __name__ == "__main__":
    unittest.main()
