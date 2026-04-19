import unittest
from base_tracker import BaseExperimentTracker


class TestBaseExperimentTracker(unittest.TestCase):
    def test_methods_exist(self):
        methods = ["start_run", "log_params", "log_metrics", "log_model", "end_run"]
        for method in methods:
            self.assertTrue(hasattr(BaseExperimentTracker, method), f"Missing {method}")

    def test_start_run_implemented(self):
        with self.assertRaises(NotImplementedError):
            BaseExperimentTracker().start_run()

    def test_log_params_implemented(self):
        with self.assertRaises(NotImplementedError):
            BaseExperimentTracker().log_params({})

    def test_log_metrics_implemented(self):
        with self.assertRaises(NotImplementedError):
            BaseExperimentTracker().log_metrics({})

    def test_log_model_implemented(self):
        with self.assertRaises(NotImplementedError):
            BaseExperimentTracker().log_model(None, "")

    def test_end_run_implemented(self):
        with self.assertRaises(NotImplementedError):
            BaseExperimentTracker().end_run()
