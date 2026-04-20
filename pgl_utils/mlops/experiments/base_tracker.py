class BaseExperimentTracker:
    def start_run(self):
        raise NotImplementedError

    def log_params(self, params: dict):
        raise NotImplementedError

    def log_metrics(self, metrics: dict):
        raise NotImplementedError

    def log_model(self, model, name: str):
        raise NotImplementedError

    def end_run(self):
        raise NotImplementedError
