"""Compatibility wrapper for the rule-based recommender."""
from __future__ import annotations
from typing import Any
from eyewear_system.recommender.rule_based_recommender import RuleBasedEyewearRecommender


class RecommenderNode:
    def __init__(self) -> None:
        self.recommender = RuleBasedEyewearRecommender()

    def recommend(self, features: dict[str, Any] | None) -> dict[str, Any]:
        return self.recommender.recommend(features)

    def recommend_sentence(self, features: dict[str, Any] | None) -> str:
        return self.recommender.recommend_sentence(features)