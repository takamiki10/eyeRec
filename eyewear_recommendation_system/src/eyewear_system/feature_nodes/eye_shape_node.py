"""Eye shape feature node backed by the local trained model."""

from typing import Any

from eyewear_system.feature_nodes.base_node import BaseFeatureNode, FeatureOutput
from eyewear_system.model_adapters.trained_feature_adapter import eye_shape_classifier


class EyeShapeNode(BaseFeatureNode):
    feature_name = "eye_shape"
    source_model = "train_eyeShape/artifacts/exported/eye_shape_model.pt"

    def predict(self, image: Any) -> FeatureOutput:
        value, confidence = eye_shape_classifier.predict(image)
        return self._format_output(value=value, confidence=confidence)
