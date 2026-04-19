from abc import ABC, abstractmethod

class BaseExperimentTracker(ABC):
    @abstractmethod
    def start_run(self):
        pass

    @abstractmethod
    def log_params(self, params: dict):
        pass

    @abstractmethod
    def log_metrics(self, metrics: dict):
        pass

    @abstractmethod
    def log_model(self, model, name: str):
        pass

    @abstractmethod
    def end_run(self):
        pass
