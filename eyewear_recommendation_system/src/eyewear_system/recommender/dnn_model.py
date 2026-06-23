"""DNN-based eyewear recommendation scorer.

The model scores each catalog item from the detected user features plus the
item's frame attributes. If a JSON checkpoint is supplied the network weights
are loaded from it; otherwise it uses deterministic initialization and clearly
marks the output as untrained.
"""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Optional, Union

from eyewear_system.recommender.ranking import sort_by_score
from eyewear_system.recommender.rule_based_recommender import (
    CLOSE_SET_THRESHOLD,
    WIDE_SET_THRESHOLD,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

FACE_SHAPES = ["oval", "round", "square", "heart", "diamond", "oblong", "triangle"]
EYE_SHAPES = ["almond", "round", "hooded", "monolid", "downturned", "upturned"]
EYE_COLORS = ["brown", "blue", "green", "hazel", "gray"]
EYE_DISTANCES = ["close_set", "average", "wide_set"]


class EyewearDNN:
    """Small feed-forward network for catalog compatibility scoring."""

    def __init__(self, input_dim: int) -> None:
        rng = random.Random(7)
        self.weights = [
            _make_weight_matrix(input_dim, 48, rng),
            _make_weight_matrix(48, 24, rng),
            _make_weight_matrix(24, 1, rng),
        ]
        self.biases = [
            _make_bias_vector(48, rng),
            _make_bias_vector(24, rng),
            _make_bias_vector(1, rng),
        ]

    def predict_score(self, inputs: list[float]) -> float:
        hidden_1 = [_relu(value) for value in _dense(inputs, self.weights[0], self.biases[0])]
        hidden_2 = [_relu(value) for value in _dense(hidden_1, self.weights[1], self.biases[1])]
        output = _dense(hidden_2, self.weights[2], self.biases[2])[0]
        return _sigmoid(output)

    def load_state(self, state: dict[str, Any]) -> None:
        self.weights = state["weights"]
        self.biases = state["biases"]


class DNNRecommendationNode:
    """Predict top catalog matches with a DNN scorer."""

    def __init__(
        self,
        catalog_path: Optional[Union[str, Path]] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.catalog_path = (
            Path(catalog_path)
            if catalog_path
            else PROJECT_ROOT / "data" / "eyewear_catalog" / "eyewear_items.csv"
        )
        self.catalog = self._load_catalog()
        self.frame_shapes = sorted({item["frame_shape"] for item in self.catalog})
        self.frame_colors = sorted({item["frame_color"] for item in self.catalog})
        self.style_tags = sorted({item["style_tag"] for item in self.catalog})
        self.input_dim = (
            len(FACE_SHAPES)
            + len(EYE_SHAPES)
            + len(EYE_COLORS)
            + len(EYE_DISTANCES)
            + 1
            + len(self.frame_shapes)
            + len(self.frame_colors)
            + len(self.style_tags)
        )

        self.model = EyewearDNN(self.input_dim)
        self.trained = False
        if checkpoint_path:
            self._load_checkpoint(Path(checkpoint_path))

    def _load_catalog(self) -> list[dict[str, str]]:
        with self.catalog_path.open("r", encoding="utf-8", newline="") as file:
            return list(csv.DictReader(file))

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        with checkpoint_path.open("r", encoding="utf-8") as file:
            checkpoint = json.load(file)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state(state_dict)
        self.trained = True

    def recommend(self, features: dict[str, Any] | None, top_k: int = 3) -> dict[str, Any]:
        """Return DNN-ranked catalog recommendations."""

        features = features or {}
        scored_items = []
        for item in self.catalog:
            vector = self._encode_features_and_item(features, item)
            score = self.model.predict_score(vector)
            scored_items.append(
                {
                    "item_id": item["item_id"],
                    "frame_shape": item["frame_shape"],
                    "frame_color": item["frame_color"],
                    "style_tag": item["style_tag"],
                    "score": round(score, 4),
                    "reason": self._build_reason(features, item),
                }
            )

        ranked = sort_by_score(scored_items)[:top_k]
        for index, item in enumerate(ranked, start=1):
            item["rank"] = index

        return {
            "success": bool(ranked),
            "model_type": "dnn",
            "trained": self.trained,
            "warning": None
            if self.trained
            else (
                "DNN scorer is running with deterministic untrained weights. "
                "Use a trained checkpoint before treating scores as learned preferences."
            ),
            "recommendations": ranked,
        }

    def _encode_features_and_item(
        self,
        features: dict[str, Any],
        item: dict[str, str],
    ) -> list[float]:
        pupil_distance = _safe_float(features.get("pupil_distance"), default=0.46)
        eye_distance = _eye_distance_from_pupil_distance(pupil_distance)

        vector: list[float] = []
        vector.extend(_one_hot(_normalize(features.get("face_shape")), FACE_SHAPES))
        vector.extend(_one_hot(_normalize(features.get("eye_shape")), EYE_SHAPES))
        vector.extend(_one_hot(_normalize(features.get("eye_color")), EYE_COLORS))
        vector.extend(_one_hot(eye_distance, EYE_DISTANCES))
        vector.append(pupil_distance)
        vector.extend(_one_hot(item["frame_shape"], self.frame_shapes))
        vector.extend(_one_hot(item["frame_color"], self.frame_colors))
        vector.extend(_one_hot(item["style_tag"], self.style_tags))
        return vector

    def _build_reason(self, features: dict[str, Any], item: dict[str, str]) -> str:
        face_shape = _normalize(features.get("face_shape")) or "unknown"
        eye_color = _normalize(features.get("eye_color")) or "unknown"
        training_state = "learned" if self.trained else "untrained"
        return (
            f"{training_state} DNN compatibility score for {item['frame_shape']} "
            f"{item['frame_color']} frames with a {face_shape} face and {eye_color} eyes"
        )


def _one_hot(value: str, vocabulary: list[str]) -> list[float]:
    return [1.0 if value == option else 0.0 for option in vocabulary]


def _normalize(value: Any) -> str:
    if value is None:
        return ""
    normalized = str(value).strip().lower()
    return normalized.replace("-", "_").replace(" ", "_")


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _eye_distance_from_pupil_distance(pupil_distance: float) -> str:
    if pupil_distance < CLOSE_SET_THRESHOLD:
        return "close_set"
    if pupil_distance > WIDE_SET_THRESHOLD:
        return "wide_set"
    return "average"


def _make_weight_matrix(
    input_dim: int,
    output_dim: int,
    rng: random.Random,
) -> list[list[float]]:
    scale = 1.0 / math.sqrt(input_dim)
    return [
        [rng.uniform(-scale, scale) for _ in range(input_dim)]
        for _ in range(output_dim)
    ]


def _make_bias_vector(output_dim: int, rng: random.Random) -> list[float]:
    return [rng.uniform(-0.05, 0.05) for _ in range(output_dim)]


def _dense(
    inputs: list[float],
    weights: list[list[float]],
    biases: list[float],
) -> list[float]:
    return [
        sum(input_value * weight for input_value, weight in zip(inputs, row)) + bias
        for row, bias in zip(weights, biases)
    ]


def _relu(value: float) -> float:
    return max(0.0, value)


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))
