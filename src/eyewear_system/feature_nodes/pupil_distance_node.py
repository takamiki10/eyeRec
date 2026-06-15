"""Placeholder pupil distance feature node."""

from typing import Any

from eyewear_system.feature_nodes.base_node import BaseFeatureNode, FeatureOutput


class PupilDistanceNode(BaseFeatureNode):
    feature_name = "pupil_distance"
    source_model = "Roboflow_Pupillary_Distance_placeholder"

    def predict(self, image: Any) -> FeatureOutput:
        return self._format_output(
            value=0.46,
            confidence=0.85,
            metadata={"unit": "normalized_face_width", "raw_px": 142.5},
        )
