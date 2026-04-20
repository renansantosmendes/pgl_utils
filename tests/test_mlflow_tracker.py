import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from pgl_utils.mlops.experiments.mlflow_tracker import MLflowTracker

_MLFLOW = "pgl_utils.mlops.experiments.mlflow_tracker.mlflow"


def _make_tracker(**kwargs) -> MLflowTracker:
    """Build an MLflowTracker with mlflow side-effects patched out."""
    with patch(f"{_MLFLOW}.set_tracking_uri"), patch(f"{_MLFLOW}.set_experiment"):
        return MLflowTracker("TestExperiment", **kwargs)


class TestRunLifecycle(unittest.TestCase):
    def setUp(self):
        self.tracker = _make_tracker()
        self.tracker.client = MagicMock()

    def test_start_run_calls_client(self):
        self.tracker.start_run()
        self.tracker.client.start_run.assert_called_once_with(
            run_name=None, tags={}
        )

    def test_start_run_with_name(self):
        self.tracker.start_run(run_name="my-run")
        self.tracker.client.start_run.assert_called_once_with(
            run_name="my-run", tags={}
        )

    def test_end_run_calls_client(self):
        self.tracker.end_run()
        self.tracker.client.end_run.assert_called_once()

    def test_context_manager_start_and_end(self):
        self.tracker.client.start_run.return_value = MagicMock()
        with self.tracker:
            self.tracker.client.start_run.assert_called_once()
        self.tracker.client.end_run.assert_called_once()


class TestRunName(unittest.TestCase):
    def test_run_name_stored_on_init(self):
        tracker = _make_tracker(run_name="exp-01")
        self.assertEqual(tracker._run_name, "exp-01")

    def test_run_name_none_by_default(self):
        tracker = _make_tracker()
        self.assertIsNone(tracker._run_name)

    def test_run_name_passed_to_start_run_via_context_manager(self):
        tracker = _make_tracker(run_name="training-v2")
        tracker.client = MagicMock()
        tracker.client.start_run.return_value = MagicMock()
        with tracker:
            tracker.client.start_run.assert_called_once_with(
                run_name="training-v2", tags={}
            )


class TestParamsAndMetrics(unittest.TestCase):
    def setUp(self):
        self.tracker = _make_tracker()
        self.tracker.client = MagicMock()

    def test_log_params(self):
        params = {"lr": 1e-3, "epochs": 10}
        self.tracker.log_params(params)
        self.tracker.client.log_params.assert_called_once_with(params)

    def test_log_metrics_without_step(self):
        metrics = {"accuracy": 0.95}
        self.tracker.log_metrics(metrics)
        self.tracker.client.log_metrics.assert_called_once_with(metrics, step=None)

    def test_log_metrics_with_step(self):
        metrics = {"loss": 0.3}
        self.tracker.log_metrics(metrics, step=5)
        self.tracker.client.log_metrics.assert_called_once_with(metrics, step=5)

    def test_log_metric_single(self):
        with patch(f"{_MLFLOW}.log_metric") as mock_metric:
            self.tracker.log_metric("val_loss", 0.42, step=3)
            mock_metric.assert_called_once_with("val_loss", 0.42, step=3)


class TestPyTorchEpochLog(unittest.TestCase):
    def setUp(self):
        self.tracker = _make_tracker()
        self.tracker.client = MagicMock()

    def test_logs_train_loss(self):
        self.tracker.pytorch_epoch_log(epoch=1, train_loss=0.5)
        self.tracker.client.log_metrics.assert_called_once_with(
            {"train_loss": 0.5}, step=1
        )

    def test_logs_val_loss(self):
        self.tracker.pytorch_epoch_log(epoch=2, train_loss=0.4, val_loss=0.45)
        metrics = self.tracker.client.log_metrics.call_args[0][0]
        self.assertIn("val_loss", metrics)
        self.assertEqual(metrics["val_loss"], 0.45)

    def test_logs_extra_metrics(self):
        self.tracker.pytorch_epoch_log(
            epoch=1, train_loss=0.3, extra_metrics={"accuracy": 0.9}
        )
        metrics = self.tracker.client.log_metrics.call_args[0][0]
        self.assertIn("accuracy", metrics)


class TestDagHubIntegration(unittest.TestCase):
    def test_dagshub_sets_tracking_uri(self):
        with patch(f"{_MLFLOW}.set_tracking_uri") as mock_uri, \
             patch(f"{_MLFLOW}.set_experiment"):
            MLflowTracker(
                "exp",
                dagshub_url="https://dagshub.com/user/myrepo",
                dagshub_user="user",
                dagshub_token="tok123",
            )
            mock_uri.assert_called_once_with(
                "https://dagshub.com/user/myrepo.mlflow"
            )

    def test_dagshub_strips_extra_path_segments(self):
        with patch(f"{_MLFLOW}.set_tracking_uri") as mock_uri, \
             patch(f"{_MLFLOW}.set_experiment"):
            MLflowTracker(
                "exp",
                dagshub_url="https://dagshub.com/user/myrepo/experiments",
                dagshub_user="user",
                dagshub_token="tok123",
            )
            mock_uri.assert_called_once_with(
                "https://dagshub.com/user/myrepo.mlflow"
            )

    def test_dagshub_sets_env_vars(self):
        env_backup = os.environ.copy()
        try:
            with patch(f"{_MLFLOW}.set_tracking_uri"), \
                 patch(f"{_MLFLOW}.set_experiment"):
                MLflowTracker(
                    "exp",
                    dagshub_url="https://dagshub.com/user/repo",
                    dagshub_user="myuser",
                    dagshub_token="mytoken",
                )
            self.assertEqual(os.environ["MLFLOW_TRACKING_USERNAME"], "myuser")
            self.assertEqual(os.environ["MLFLOW_TRACKING_PASSWORD"], "mytoken")
        finally:
            os.environ.clear()
            os.environ.update(env_backup)

    def test_no_dagshub_uses_local_uri(self):
        with patch(f"{_MLFLOW}.set_tracking_uri") as mock_uri, \
             patch(f"{_MLFLOW}.set_experiment"):
            MLflowTracker("exp", tracking_uri="./mlruns")
            mock_uri.assert_called_once_with("./mlruns")

    def test_partial_dagshub_params_falls_back_to_local(self):
        # Only url provided — should NOT trigger DagHub setup
        with patch(f"{_MLFLOW}.set_tracking_uri") as mock_uri, \
             patch(f"{_MLFLOW}.set_experiment"):
            MLflowTracker(
                "exp",
                tracking_uri="./mlruns",
                dagshub_url="https://dagshub.com/user/repo",
            )
            mock_uri.assert_called_once_with("./mlruns")


class TestLogImage(unittest.TestCase):
    def setUp(self):
        self.tracker = _make_tracker()

    def test_log_image_from_png_path(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name
        try:
            with patch(f"{_MLFLOW}.log_artifact") as mock_artifact:
                self.tracker.log_image(tmp_path, artifact_path="plots/img.png")
                mock_artifact.assert_called_once()
        finally:
            os.unlink(tmp_path)

    def test_log_image_from_gif_path(self):
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
            tmp_path = f.name
        try:
            with patch(f"{_MLFLOW}.log_artifact") as mock_artifact:
                self.tracker.log_image(tmp_path, artifact_path="plots/anim.gif")
                mock_artifact.assert_called_once()
        finally:
            os.unlink(tmp_path)

    def test_log_image_invalid_extension_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".bmp", delete=False) as f:
            tmp_path = f.name
        try:
            with self.assertRaises(ValueError):
                self.tracker.log_image(tmp_path, artifact_path="plots/img.bmp")
        finally:
            os.unlink(tmp_path)

    def test_log_image_from_numpy_array(self):
        import numpy as np
        arr = np.zeros((32, 32, 3), dtype=np.uint8)
        with patch(f"{_MLFLOW}.log_image") as mock_log_image:
            self.tracker.log_image(arr, artifact_path="plots/img.png")
            mock_log_image.assert_called_once_with(arr, "plots/img.png", step=None)

    def test_log_image_from_numpy_with_step(self):
        import numpy as np
        arr = np.zeros((16, 16, 3), dtype=np.uint8)
        with patch(f"{_MLFLOW}.log_image") as mock_log_image:
            self.tracker.log_image(arr, artifact_path="plots/img.png", step=10)
            mock_log_image.assert_called_once_with(arr, "plots/img.png", step=10)


if __name__ == "__main__":
    unittest.main()
