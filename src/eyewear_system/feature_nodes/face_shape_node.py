"""Placeholder face shape feature node."""

from typing import Any

from eyewear_system.feature_nodes.base_node import BaseFeatureNode, FeatureOutput


class FaceShapeNode(BaseFeatureNode):
    feature_name = "face_shape"
    source_model = "Roboflow_Face_Shape_placeholder"

    def predict(self, image: Any) -> FeatureOutput:
        return self._format_output(value="oval", confidence=0.91)
