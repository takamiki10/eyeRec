from eyewear_system.feature_nodes import (
    EyeColorNode,
    EyeShapeNode,
    FaceShapeNode,
    PupilDistanceNode,
)
from eyewear_system.feature_nodes import eye_color_node
from PIL import Image


STANDARD_KEYS = {"feature_name", "value", "confidence", "source_model", "metadata"}


def test_feature_nodes_return_standard_keys():
    image = Image.new("RGB", (32, 32), color=(128, 96, 64))
    for node in [EyeColorNode(), EyeShapeNode(), PupilDistanceNode(), FaceShapeNode()]:
        output = node.predict(image=image)
        assert STANDARD_KEYS == set(output.keys())


def test_low_confidence_gray_eye_color_maps_to_brown(monkeypatch):
    monkeypatch.setattr(
        eye_color_node.eye_color_classifier,
        "predict",
        lambda image: ("gray", 0.42),
    )

    output = EyeColorNode().predict(image=None)

    assert output["value"] == "brown"
    assert output["confidence"] == 0.42
