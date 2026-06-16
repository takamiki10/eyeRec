"""Full placeholder recommendation pipeline."""

from pathlib import Path

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

    def run(self, image_path: str | Path) -> dict:
        image_path = Path(image_path)
        image = load_image(image_path)
        node_outputs = [node.predict(image) for node in self.feature_nodes]
        aggregated_features = self.aggregator.aggregate(node_outputs)
        recommendations = self.recommender.recommend(aggregated_features)
        return {
            "input_image": str(image_path),
            "node_outputs": node_outputs,
            "aggregated_features": aggregated_features,
            "recommendations": recommendations,
        }
