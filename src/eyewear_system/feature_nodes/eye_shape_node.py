"""Placeholder eye shape feature node."""

from typing import Any

from eyewear_system.feature_nodes.base_node import BaseFeatureNode, FeatureOutput


class EyeShapeNode(BaseFeatureNode):
    feature_name = "eye_shape"
    source_model = "Roboflow_Eye_Shapes_placeholder"

    def predict(self, image: Any) -> FeatureOutput:
        return self._format_output(value="almond", confidence=0.88)
