"""Full eyewear recommendation pipeline."""

from pathlib import Path
from typing import Any, Union

from eyewear_system.feature_nodes import (
    EyeColorNode,
    EyeShapeNode,
    FaceShapeNode,
    PupilDistanceNode,
)
from eyewear_system.pipeline.feature_aggregator import FeatureAggregator
from eyewear_system.recommender.recommender_node import RecommenderNode
from eyewear_system.utils.image_io import load_image


class EyewearRecommendationPipeline:
    """Run placeholder feature extraction and recommendation end to end."""

    def __init__(self) -> None:
        self.feature_nodes = [
            EyeColorNode(),
            EyeShapeNode(),
            PupilDistanceNode(),
            FaceShapeNode(),
        ]
        self.aggregator = FeatureAggregator()
        self.recommender = RecommenderNode()

    def run(self, image_path: Union[str, Path], include_debug: bool = False) -> dict:
        image_path = Path(image_path)
        image = load_image(image_path)
        node_outputs = [node.predict(image) for node in self.feature_nodes]
        aggregated_features = self.aggregator.aggregate(node_outputs)
        recommendations = self.recommender.recommend(aggregated_features)

        result = {
            "input_image": str(image_path),
            "detected_features": _simplify_features(aggregated_features),
            "rule_based": recommendations["rule_based"],
            "dnn": recommendations["dnn"],
        }
        if include_debug:
            result["debug"] = {
                "node_outputs": node_outputs,
                "aggregated_features": aggregated_features,
            }
        return result


def _simplify_features(features: dict[str, Any]) -> dict[str, Any]:
    confidences = features.get("confidences", {})
    simplified = {}
    for feature_name in ("face_shape", "eye_shape", "eye_color", "pupil_distance"):
        value = features.get(feature_name)
        confidence = confidences.get(feature_name)
        simplified[feature_name] = {
            "value": value,
            "confidence": round(float(confidence), 3) if confidence is not None else None,
        }
    return simplified
