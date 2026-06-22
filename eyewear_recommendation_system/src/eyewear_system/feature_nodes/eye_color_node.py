"""Eye color feature node backed by the local trained model."""

from typing import Any

from eyewear_system.feature_nodes.base_node import BaseFeatureNode, FeatureOutput
from eyewear_system.model_adapters.trained_feature_adapter import eye_color_classifier


class EyeColorNode(BaseFeatureNode):
    feature_name = "eye_color"
    source_model = "train_eyeColor/artifacts/exported/eye_color_model.pt"

    def predict(self, image: Any) -> FeatureOutput:
        value, confidence = eye_color_classifier.predict(image)
        return self._format_output(value=value, confidence=confidence)
