"""Feature aggregation utilities."""

from eyewear_system.feature_nodes.base_node import FeatureOutput


class FeatureAggregator:
    """Convert standard node outputs into recommender-ready features."""

    def aggregate(self, node_outputs: list[FeatureOutput]) -> dict:
        aggregated = {"confidences": {}}
        for output in node_outputs:
            feature_name = output["feature_name"]
            aggregated[feature_name] = output["value"]
            aggregated["confidences"][feature_name] = output["confidence"]
        return aggregated
