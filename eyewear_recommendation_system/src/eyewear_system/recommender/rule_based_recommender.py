"""Rule-based eyewear recommender

This module expects already-detected facial features as input and maps them to
a human-readable eyewear recommendation using hard-coded rule tables.

Expected input:
{
    "face_shape": "round",
    "eye_shape": "almond",
    "eye_color": "brown",
    "pupil_distance": 0.46
}
"""

from __future__ import annotations
import math
from dataclasses import asdict, dataclass
from typing import Any


MISSING_INPUT_WARNING = (
    "Recommendation could not be generated because one or more required "
    "input features are missing or invalid."
)

# ---------------------------------------------------------------------------
# TODO_REPLACE_LATER:
# These thresholds should be adjusted after seeing real pupil-distance outputs
# from the trained model.
#
# Current interpretation:
# pupil_distance < 0.43  -> close_set
# 0.43 - 0.50            -> average
# pupil_distance > 0.50  -> wide_set
# ---------------------------------------------------------------------------
CLOSE_SET_THRESHOLD = 0.43
WIDE_SET_THRESHOLD = 0.50

# rule tables
FACE_SHAPE_RULES = {
    "oval": {
        "shape": ["rectangular", "square", "geometric", "subtle cat-eye"],
        "avoid": ["very narrow frames"],
    },
    "round": {
        "shape": ["rectangular", "square", "wayfarer", "angular cat-eye"],
        "avoid": ["round frames"],
    },
    "square": {
        "shape": ["round", "oval", "thin metal", "soft cat-eye"],
        "avoid": ["very boxy square frames"],
    },
    "heart": {
        "shape": ["oval", "round", "light rectangular", "rimless", "bottom-heavy"],
        "avoid": ["very heavy top frames"],
    },
    "diamond": {
        "shape": ["oval", "cat-eye", "browline", "rimless"],
        "avoid": ["very narrow frames", "boxy frames"],
    },
    "oblong": {
        "shape": ["tall rectangular", "oversized", "aviator", "round"],
        "avoid": ["tiny frames", "very narrow frames"],
    },
    "triangle": {
        "shape": ["browline", "cat-eye", "aviator", "top-heavy rectangular"],
        "avoid": ["heavy bottom frames"],
    },
}

EYE_SHAPE_MODIFIERS = {
    "almond": "balanced or slightly upswept frame",
    "round": "more angular or rectangular lenses",
    "hooded": "visible upper rim, browline, or subtle cat-eye",
    "monolid": "clean geometric, rectangular, or oval frame with nose pads",
    "downturned": "cat-eye, upswept rectangular, or lifted browline frame",
    "upturned": "balanced oval, rectangular, or softly rounded frame",
}

EYE_DISTANCE_MODIFIERS = {
    "close_set": "narrow bridge, lighter or clear bridge",
    "average": "standard bridge and balanced frame width",
    "wide_set": "wider bridge and stronger center bridge",
}

EYE_COLOR_RULES = {
    "brown": ["tortoiseshell", "gold", "green", "black", "warm brown"],
    "blue": ["silver", "gray", "blue", "navy", "cool transparent"],
    "green": ["purple", "plum", "burgundy", "brown", "emerald", "tortoiseshell"],
    "hazel": ["tortoiseshell", "bronze", "green", "gray", "warm neutral"],
    "gray": ["navy", "silver", "gunmetal", "black", "burgundy", "purple"],
}

CATEGORY_ALIASES = {
    "grey": "gray",
    "pear": "triangle",
    "pear_shaped": "triangle",
    "triangle_face": "triangle",
}


@dataclass
class EyewearRecommendation:
    """Structured recommendation output."""

    success: bool
    face_shape: str
    eye_shape: str
    eye_color: str
    eye_distance: str
    pupil_distance: float
    recommended_shapes: list[str]
    recommended_colors: list[str]
    eye_shape_modifier: str
    eye_distance_modifier: str
    avoid: list[str]
    sentence: str
    warning: str | None = None


class RuleBasedEyewearRecommender:
    """Hard-coded eyewear recommender.
    The neural networks detect the input features.
    This recommender only applies rule tables and creates a sentence.
    """

    def recommend(self, features: dict[str, Any] | None) -> dict[str, Any]:
        """Return a structured recommendation dictionary.
        If required input features are missing, the method returns a warning.
        """

        features = features or {}

        if self._has_missing_input(features):
            return self._warning_response()

        face_shape = _normalize_category(features["face_shape"])
        eye_shape = _normalize_category(features["eye_shape"])
        eye_color = _normalize_category(features["eye_color"])

        pupil_distance = self._parse_pupil_distance(features["pupil_distance"])
        if pupil_distance is None:
            return self._warning_response()

        eye_distance = self._convert_pupil_distance_to_eye_distance(pupil_distance)

        if face_shape not in FACE_SHAPE_RULES:
            return self._warning_response()

        if eye_shape not in EYE_SHAPE_MODIFIERS:
            return self._warning_response()

        if eye_color not in EYE_COLOR_RULES:
            return self._warning_response()

        face_rule = FACE_SHAPE_RULES[face_shape]
        recommended_shapes = face_rule["shape"]
        avoid = face_rule["avoid"]

        recommended_colors = EYE_COLOR_RULES[eye_color]
        eye_shape_modifier = EYE_SHAPE_MODIFIERS[eye_shape]
        eye_distance_modifier = EYE_DISTANCE_MODIFIERS[eye_distance]

        sentence = self._build_sentence(
            face_shape=face_shape,
            eye_shape=eye_shape,
            eye_color=eye_color,
            eye_distance=eye_distance,
            recommended_shapes=recommended_shapes,
            recommended_colors=recommended_colors,
            eye_shape_modifier=eye_shape_modifier,
            eye_distance_modifier=eye_distance_modifier,
            avoid=avoid,
        )

        recommendation = EyewearRecommendation(
            success=True,
            face_shape=face_shape,
            eye_shape=eye_shape,
            eye_color=eye_color,
            eye_distance=eye_distance,
            pupil_distance=pupil_distance,
            recommended_shapes=recommended_shapes,
            recommended_colors=recommended_colors,
            eye_shape_modifier=eye_shape_modifier,
            eye_distance_modifier=eye_distance_modifier,
            avoid=avoid,
            sentence=sentence,
            warning=None,
        )

        return asdict(recommendation)

    def recommend_sentence(self, features: dict[str, Any] | None) -> str:
        """Return only the final sentence or the warning text."""
        result = self.recommend(features)
        if not result["success"]:
            return result["warning"]
        return result["sentence"]

    def _has_missing_input(self, features: dict[str, Any]) -> bool:
        """Check whether required input features are missing."""
        required_features = [
            "face_shape",
            "eye_shape",
            "eye_color",
            "pupil_distance",
        ]
        for feature_name in required_features:
            if _is_missing(features.get(feature_name)):
                return True
        return False

    def _parse_pupil_distance(self, pupil_distance: Any) -> float | None:
        """Convert the pupil distance input to a valid float."""
        try:
            value = float(pupil_distance)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        return value

    def _convert_pupil_distance_to_eye_distance(self, pupil_distance: float) -> str:
        """Convert numeric pupil distance into an eye-distance category."""
        if pupil_distance < CLOSE_SET_THRESHOLD:
            return "close_set"
        if pupil_distance > WIDE_SET_THRESHOLD:
            return "wide_set"
        return "average"

    def _warning_response(self) -> dict[str, Any]:
        """Return a consistent warning output."""
        return {
            "success": False,
            "warning": MISSING_INPUT_WARNING,
            "sentence": None,
            "face_shape": None,
            "eye_shape": None,
            "eye_color": None,
            "eye_distance": None,
            "pupil_distance": None,
            "recommended_shapes": [],
            "recommended_colors": [],
            "eye_shape_modifier": None,
            "eye_distance_modifier": None,
            "avoid": [],
        }

    def _build_sentence(
        self,
        face_shape: str,
        eye_shape: str,
        eye_color: str,
        eye_distance: str,
        recommended_shapes: list[str],
        recommended_colors: list[str],
        eye_shape_modifier: str,
        eye_distance_modifier: str,
        avoid: list[str],
    ) -> str:
        face_article = _article_for(face_shape)

        shape_text = _format_list(recommended_shapes)
        color_text = _format_list(recommended_colors)
        avoid_text = _format_list(avoid)

        return (
            f"For {face_article} {face_shape} face with {eye_shape} {eye_color} eyes, "
            f"{shape_text} frames in {color_text} would usually fit well. "
            f"Because the detected eye shape is {eye_shape}, a {eye_shape_modifier} is recommended. "
            f"For {_humanize(eye_distance)} eyes, choose a {eye_distance_modifier}. "
            f"It is better to avoid {avoid_text}."
        )


def _normalize_category(value: Any) -> str:
    """Normalize model output strings.
    Examples:
        "Grey" -> "gray"
        "pear-shaped" -> "triangle"
        "Round" -> "round"
    """

    if value is None:
        return ""

    normalized = str(value).strip().lower()
    normalized = normalized.replace("-", "_").replace(" ", "_")

    return CATEGORY_ALIASES.get(normalized, normalized)


def _is_missing(value: Any) -> bool:
    """Return True if an input value is missing or empty."""
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _format_list(values: list[str]) -> str:
    """Convert a list into a natural English phrase."""

    if not values:
        return ""

    if len(values) == 1:
        return values[0]

    if len(values) == 2:
        return f"{values[0]} or {values[1]}"

    return f"{', '.join(values[:-1])}, or {values[-1]}"


def _article_for(word: str) -> str:
    """Return a/an for simple English sentence generation."""
    if word[:1].lower() in {"a", "e", "i", "o", "u"}:
        return "an"
    return "a"


def _humanize(value: str) -> str:
    """Convert internal labels into readable text."""
    return value.replace("_", " ")