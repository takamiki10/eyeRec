"""Placeholder recommender node."""


class RecommenderNode:
    """Dummy rule-based recommender returning a fixed top 3."""

    def recommend(self, features: dict) -> list[dict]:
        return [
            {
                "rank": 1,
                "frame_shape": "round",
                "frame_color": "dark brown",
                "score": 0.92,
                "reason": "Placeholder recommendation based on oval face shape and brown eye color.",
            },
            {
                "rank": 2,
                "frame_shape": "soft square",
                "frame_color": "black",
                "score": 0.87,
                "reason": "Placeholder recommendation.",
            },
            {
                "rank": 3,
                "frame_shape": "oval",
                "frame_color": "tortoise",
                "score": 0.83,
                "reason": "Placeholder recommendation.",
            },
        ]
