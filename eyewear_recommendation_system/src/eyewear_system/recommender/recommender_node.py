"""Rule-based eyewear recommender node."""

import csv
from pathlib import Path
from typing import Optional, Union


PROJECT_ROOT = Path(__file__).resolve().parents[3]

FACE_SHAPE_FRAME_MATCHES = {
    "oval": {
        "round", "soft square", "rectangle", "cat eye", "aviator", "browline", "oval", "square",
        "wayfarer", "geometric", "hexagon", "panto", "rimless", "clubmaster", "navigator",
    },
    "round": {
        "rectangle", "square", "browline", "soft square", "wayfarer", "geometric",
        "d frame", "flat top", "sport rectangle", "navigator",
    },
    "square": {
        "round", "oval", "aviator", "panto", "rimless", "semi rimless", "rounded square",
        "thin metal", "butterfly",
    },
    "heart": {
        "round", "oval", "cat eye", "butterfly", "rimless", "semi rimless", "panto",
        "keyhole bridge", "thin metal",
    },
    "oblong": {
        "round", "aviator", "oval", "browline", "oversized", "wayfarer", "clubmaster",
        "double bridge", "navigator", "chunky acetate",
    },
}

EYE_COLOR_MATCHES = {
    "brown": {"dark brown", "tortoise", "gold", "black", "espresso", "walnut", "crystal brown", "amber", "burgundy"},
    "blue": {"black", "gunmetal", "navy", "clear", "matte black", "silver", "champagne"},
    "green": {"forest green", "tortoise", "burgundy", "gold", "teal", "plum"},
    "hazel": {"tortoise", "gold", "forest green", "dark brown", "amber", "walnut", "crystal brown"},
    "gray": {"black", "gunmetal", "clear", "navy", "matte black", "silver", "champagne"},
    "grey": {"black", "gunmetal", "clear", "navy", "matte black", "silver", "champagne"},
}


class RecommenderNode:
    """Small deterministic recommender using extracted face and eye features."""

    def __init__(self, catalog_path: Optional[Union[str, Path]] = None) -> None:
        self.catalog_path = Path(catalog_path) if catalog_path else PROJECT_ROOT / "data" / "eyewear_catalog" / "eyewear_items.csv"
        self.catalog = self._load_catalog()

    def _load_catalog(self) -> list[dict]:
        with self.catalog_path.open("r", encoding="utf-8", newline="") as file:
            return list(csv.DictReader(file))

    def recommend(self, features: dict) -> list[dict]:
        face_shape = str(features.get("face_shape", "")).lower()
        eye_color = str(features.get("eye_color", "")).lower()
        confidences = features.get("confidences", {})
        face_confidence = float(confidences.get("face_shape", 0.0))
        eye_confidence = float(confidences.get("eye_color", 0.0))

        preferred_shapes = FACE_SHAPE_FRAME_MATCHES.get(face_shape, set())
        preferred_colors = EYE_COLOR_MATCHES.get(eye_color, set())
        scored_items = []

        for item in self.catalog:
            score = 0.50
            reasons = []

            if item["frame_shape"] in preferred_shapes:
                score += 0.30 * max(face_confidence, 0.35)
                reasons.append(f"{item['frame_shape']} frames balance a {face_shape} face")
            if item["frame_color"] in preferred_colors:
                score += 0.25 * max(eye_confidence, 0.35)
                reasons.append(f"{item['frame_color']} complements {eye_color} eyes")
            if item["style_tag"] in {"classic", "professional", "minimal"}:
                score += 0.03

            scored_items.append(
                {
                    "frame_shape": item["frame_shape"],
                    "frame_color": item["frame_color"],
                    "style_tag": item["style_tag"],
                    "score": round(min(score, 0.99), 4),
                    "reason": "; ".join(reasons) if reasons else f"General fallback match for {face_shape} face and {eye_color} eyes",
                }
            )

        ranked = sorted(scored_items, key=lambda item: item["score"], reverse=True)[:3]
        for index, item in enumerate(ranked, start=1):
            item["rank"] = index
        return ranked
