"""Combined eyewear recommender node."""

from __future__ import annotations

from typing import Any

from eyewear_system.recommender.dnn_model import DNNRecommendationNode
from eyewear_system.recommender.rule_based_recommender import RuleBasedEyewearRecommender


class RecommenderNode:
    """Run rule-based and DNN recommenders from one entry point."""

    def __init__(self) -> None:
        self.rule_based_recommender = RuleBasedEyewearRecommender()
        self.dnn_recommender = DNNRecommendationNode()

    def recommend(self, features: dict[str, Any] | None) -> dict[str, Any]:
        rule_based = self.rule_based_recommender.recommend(features)
        dnn = self.dnn_recommender.recommend(features)
        return {
            "rule_based": _simplify_rule_based(rule_based),
            "dnn": _simplify_dnn(dnn),
        }

    def recommend_rule_based(self, features: dict[str, Any] | None) -> dict[str, Any]:
        """Return the full rule-based output for tests and detailed debugging."""
        return self.rule_based_recommender.recommend(features)

    def recommend_dnn(self, features: dict[str, Any] | None) -> dict[str, Any]:
        """Return the full DNN output for tests and detailed debugging."""
        return self.dnn_recommender.recommend(features)

    def recommend_sentence(self, features: dict[str, Any] | None) -> str:
        return self.rule_based_recommender.recommend_sentence(features)


def _simplify_rule_based(result: dict[str, Any]) -> dict[str, Any]:
    if not result.get("success"):
        return {
            "success": False,
            "summary": result.get("warning"),
        }

    return {
        "success": True,
        "summary": (
            f"Try {_join_list(result['recommended_shapes'])} frames in "
            f"{_join_list(result['recommended_colors'])}. "
            f"Avoid {_join_list(result['avoid'])}."
        ),
        "best_shapes": result["recommended_shapes"],
        "best_colors": result["recommended_colors"],
        "bridge_fit": result["eye_distance_modifier"],
        "avoid": result["avoid"],
    }


def _simplify_dnn(result: dict[str, Any]) -> dict[str, Any]:
    top_picks = [
        {
            "rank": item["rank"],
            "frame": f"{item['frame_shape']} / {item['frame_color']}",
            "style": item["style_tag"],
        }
        for item in result.get("recommendations", [])
    ]
    simplified = {
        "success": result.get("success", False),
        "trained": result.get("trained", False),
        "top_picks": top_picks,
    }
    return simplified


def _join_list(values: list[str]) -> str:
    if not values:
        return "none"
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} or {values[1]}"
    return f"{', '.join(values[:-1])}, or {values[-1]}"
