from eyewear_system.feature_nodes import (
    EyeColorNode,
    EyeShapeNode,
    FaceShapeNode,
    PupilDistanceNode,
)
from eyewear_system.pipeline.feature_aggregator import FeatureAggregator


def test_feature_aggregation_returns_expected_keys():
    node_outputs = [
        EyeColorNode().predict(None),
        EyeShapeNode().predict(None),
        PupilDistanceNode().predict(None),
        FaceShapeNode().predict(None),
    ]
    aggregated = FeatureAggregator().aggregate(node_outputs)

    assert aggregated == {
        "eye_color": "brown",
        "eye_shape": "almond",
        "pupil_distance": 0.24,
        "face_shape": "oval",
        "confidences": {
            "eye_color": 0.90,
            "eye_shape": 0.88,
            "pupil_distance": 0.85,
            "face_shape": 0.91,
        },
    }
