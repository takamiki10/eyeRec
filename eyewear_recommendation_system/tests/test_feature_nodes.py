from eyewear_system.feature_nodes import (
    EyeColorNode,
    EyeShapeNode,
    FaceShapeNode,
    PupilDistanceNode,
)


STANDARD_KEYS = {"feature_name", "value", "confidence", "source_model", "metadata"}


def test_feature_nodes_return_standard_keys():
    for node in [EyeColorNode(), EyeShapeNode(), PupilDistanceNode(), FaceShapeNode()]:
        output = node.predict(image=None)
        assert STANDARD_KEYS == set(output.keys())
