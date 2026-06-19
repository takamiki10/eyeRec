from pathlib import Path

import pytest

from train_pupil_distance.datasets import PupilDistanceDataset


def test_dataset_loader_raises_clear_error_when_labels_missing(tmp_path: Path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Labels CSV not found"):
        PupilDistanceDataset(image_dir, tmp_path / "labels.csv")
