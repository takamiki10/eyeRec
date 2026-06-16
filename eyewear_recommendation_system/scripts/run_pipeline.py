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
    parser = argparse.ArgumentParser(description="Run placeholder eyewear recommendations.")
    parser.add_argument("--image", required=True, help="Path to an input face image.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = EyewearRecommendationPipeline()
    result = pipeline.run(args.image)

    output_path = PROJECT_ROOT / "outputs" / "recommendations" / "top3_recommendations.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print(json.dumps(result, indent=2))
    print(f"Saved recommendations to {output_path}")


if __name__ == "__main__":
    main()
