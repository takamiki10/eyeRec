from pathlib import Path

from train_eye_shape.utils import load_label_mapping


def test_label_mapping_loads_correctly():
    project_dir = Path(__file__).resolve().parents[1]
    mapping = load_label_mapping(project_dir / "configs" / "label_mapping.yaml")

    assert mapping == {
        "almond": 0,
        "round": 1,
        "monolid": 2,
        "hooded": 3,
        "upturned": 4,
        "downturned": 5,
    }
