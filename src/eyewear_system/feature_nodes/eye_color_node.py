"""Placeholder eye color feature node."""

from typing import Any

from eyewear_system.feature_nodes.base_node import BaseFeatureNode, FeatureOutput


class EyeColorNode(BaseFeatureNode):
    feature_name = "eye_color"
    source_model = "Eye_Color_Detection_placeholder"

    def predict(self, image: Any) -> FeatureOutput:
        return self._format_output(value="brown", confidence=0.90)
