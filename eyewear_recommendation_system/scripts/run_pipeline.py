"""Run the full eyewear recommendation pipeline."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eyewear_system.pipeline.full_pipeline import EyewearRecommendationPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run eyewear recommendations.")
    parser.add_argument("--image", required=True, help="Path to an input face image.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the compact recommendation JSON instead of the readable summary.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Include raw feature-node outputs in the saved and printed JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = EyewearRecommendationPipeline()
    result = pipeline.run(args.image, include_debug=args.debug)

    output_path = PROJECT_ROOT / "outputs" / "recommendations" / "top3_recommendations.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(_format_readable_result(result))
    print(f"Saved recommendations to {output_path}")


def _format_readable_result(result: dict) -> str:
    features = result["detected_features"]
    rule_based = result["rule_based"]
    dnn = result["dnn"]

    lines = [
        "Eyewear Recommendation",
        f"Image: {result['input_image']}",
        "",
        "Detected features:",
    ]
    for label in ("face_shape", "eye_shape", "eye_color", "pupil_distance"):
        feature = features[label]
        lines.append(
            f"- {label.replace('_', ' ')}: {_format_value(feature['value'])} "
            f"(confidence {feature['confidence']})"
        )

    lines.extend(
        [
            "",
            "Rule-based:",
            f"- Best shapes: {_join_list(rule_based.get('best_shapes', []))}",
            f"- Best colors: {_join_list(rule_based.get('best_colors', []))}",
            f"- Bridge fit: {rule_based.get('bridge_fit')}",
            f"- Avoid: {_join_list(rule_based.get('avoid', []))}",
            f"- Summary: {rule_based.get('summary')}",
            "",
            "DNN:",
            f"- Trained: {dnn.get('trained')}",
            "- Top picks:",
        ]
    )
    for item in dnn.get("top_picks", []):
        lines.append(
            f"  {item['rank']}. {item['frame']} "
            f"({item['style']}, score {item['score']})"
        )
    if dnn.get("note"):
        lines.append(f"- Note: {dnn['note']}")
    return "\n".join(lines)


def _join_list(values: list) -> str:
    if not values:
        return "none"
    return ", ".join(str(value) for value in values)


def _format_value(value: object) -> object:
    if isinstance(value, float):
        return round(value, 3)
    return value


if __name__ == "__main__":
    main()
