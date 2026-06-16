"""Run one placeholder feature node."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eyewear_system.feature_nodes import (
    EyeColorNode,
    EyeShapeNode,
    FaceShapeNode,
    PupilDistanceNode,
)
from eyewear_system.utils.image_io import load_image


NODE_REGISTRY = {
    "eye_color": EyeColorNode,
    "eye_shape": EyeShapeNode,
    "pupil_distance": PupilDistanceNode,
    "face_shape": FaceShapeNode,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one placeholder feature node.")
    parser.add_argument("--node", required=True, choices=sorted(NODE_REGISTRY))
    parser.add_argument("--image", required=True, help="Path to an input face image.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = load_image(args.image)
    node = NODE_REGISTRY[args.node]()
    result = node.predict(image)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
