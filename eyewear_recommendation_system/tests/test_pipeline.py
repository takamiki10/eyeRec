from pathlib import Path

from PIL import Image

from eyewear_system.pipeline.full_pipeline import EyewearRecommendationPipeline


def test_full_pipeline_returns_expected_sections(tmp_path: Path):
    image_path = tmp_path / "example.jpg"
    Image.new("RGB", (32, 32), color=(128, 96, 64)).save(image_path)

    result = EyewearRecommendationPipeline().run(image_path)

    assert "node_outputs" in result
    assert "aggregated_features" in result
    assert "recommendations" in result
    assert len(result["node_outputs"]) == 4
    assert len(result["recommendations"]) == 3
