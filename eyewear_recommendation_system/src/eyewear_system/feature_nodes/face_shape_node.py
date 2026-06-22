"""Face shape feature node backed by the local trained model."""

from typing import Any

from eyewear_system.feature_nodes.base_node import BaseFeatureNode, FeatureOutput
from eyewear_system.model_adapters.trained_feature_adapter import face_shape_classifier


class FaceShapeNode(BaseFeatureNode):
    feature_name = "face_shape"
    source_model = "train_faceShape/artifacts/exported/face_shape_model.pt"

    def predict(self, image: Any) -> FeatureOutput:
        value, confidence = face_shape_classifier.predict(image)
        return self._format_output(value=value, confidence=confidence)
