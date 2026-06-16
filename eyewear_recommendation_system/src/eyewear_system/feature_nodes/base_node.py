"""Base class for all feature extraction nodes."""

from abc import ABC, abstractmethod
from typing import Any, Dict


FeatureOutput = Dict[str, Any]


class BaseFeatureNode(ABC):
    """Common interface for feature nodes."""

    feature_name: str
    source_model: str

    @abstractmethod
    def predict(self, image: Any) -> FeatureOutput:
        """Return a standard feature output dictionary."""
        raise NotImplementedError

    def _format_output(
        self,
        value: Any,
        confidence: float,
        metadata: dict | None = None,
    ) -> FeatureOutput:
        return {
            "feature_name": self.feature_name,
            "value": value,
            "confidence": confidence,
            "source_model": self.source_model,
            "metadata": metadata or {},
        }
