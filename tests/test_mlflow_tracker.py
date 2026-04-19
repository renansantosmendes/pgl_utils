import unittest
from pgl_utils.mlops.experiments.mlflow_tracker import MLflowTracker


class MockMLflow:
    def __init__(self):
        self.actions = []

    def start_run(self, **kwargs):
        self.actions.append("start")

    def log_params(self, params):
        self.actions.append(("params", params))

    def log_metrics(self, metrics, step=None):
        self.actions.append(("metrics", metrics))

    def log_model(self, artifact_path, python_model):
        self.actions.append(("model", artifact_path))

    def end_run(self, **kwargs):
        self.actions.append("end")


class TestMLflowTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = MLflowTracker("TestExperiment")
        self.tracker.client = MockMLflow()

    def test_start_run(self):
        self.tracker.start_run()
        self.assertIn("start", self.tracker.client.actions)

    def test_log_params(self):
        params = {"param1": 1}
        self.tracker.log_params(params)
        self.assertIn(("params", params), self.tracker.client.actions)

    def test_log_metrics(self):
        metrics = {"metric1": 0.99}
        self.tracker.log_metrics(metrics)
        self.assertIn(("metrics", metrics), self.tracker.client.actions)

    def test_log_model(self):
        class DummyModel:
            pass

        self.tracker.log_model(DummyModel(), "model_name")
        self.assertIn(("model", "model_name"), self.tracker.client.actions)

    def test_end_run(self):
        self.tracker.end_run()
        self.assertIn("end", self.tracker.client.actions)
