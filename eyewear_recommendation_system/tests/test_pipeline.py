from pathlib import Path

from PIL import Image

from eyewear_system.pipeline.full_pipeline import EyewearRecommendationPipeline


def test_full_pipeline_returns_expected_sections(tmp_path: Path):
    image_path = tmp_path / "example.jpg"
    Image.new("RGB", (32, 32), color=(128, 96, 64)).save(image_path)

    result = EyewearRecommendationPipeline().run(image_path)

    assert "detected_features" in result
    assert "rule_based" in result
    assert "dnn" in result
    assert "debug" not in result
    assert result["rule_based"]["success"] is True
    assert isinstance(result["rule_based"]["summary"], str)
    assert result["dnn"]["success"] is True
    assert len(result["dnn"]["top_picks"]) == 3


def test_full_pipeline_can_include_debug_sections(tmp_path: Path):
    image_path = tmp_path / "example.jpg"
    Image.new("RGB", (32, 32), color=(128, 96, 64)).save(image_path)

    result = EyewearRecommendationPipeline().run(image_path, include_debug=True)

    assert "debug" in result
    assert "node_outputs" in result["debug"]
    assert "aggregated_features" in result["debug"]
    assert len(result["debug"]["node_outputs"]) == 4
