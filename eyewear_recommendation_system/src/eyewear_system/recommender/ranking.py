"""Ranking helpers for future recommender implementations."""


def sort_by_score(recommendations: list[dict]) -> list[dict]:
    return sorted(recommendations, key=lambda item: item.get("score", 0.0), reverse=True)
