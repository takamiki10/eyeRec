from pathlib import Path

from train_eye_color.utils import load_label_mapping


def test_label_mapping_loads_correctly():
    project_dir = Path(__file__).resolve().parents[1]
    mapping = load_label_mapping(project_dir / "configs" / "label_mapping.yaml")

    assert mapping == {
        "brown": 0,
        "blue": 1,
        "green": 2,
        "hazel": 3,
        "grey": 4,
    }
