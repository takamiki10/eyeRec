"""Pupil distance feature node backed by the local trained model."""

from typing import Any

from eyewear_system.feature_nodes.base_node import BaseFeatureNode, FeatureOutput
from eyewear_system.model_adapters.trained_feature_adapter import pupil_distance_regressor


class PupilDistanceNode(BaseFeatureNode):
    feature_name = "pupil_distance"
    source_model = "train_pupilDistance/artifacts/exported/pupil_distance_model.pt"

    def predict(self, image: Any) -> FeatureOutput:
        value, confidence = pupil_distance_regressor.predict(image)
        return self._format_output(
            value=value,
            confidence=confidence,
            metadata={"unit": "normalized_face_width"},
        )
