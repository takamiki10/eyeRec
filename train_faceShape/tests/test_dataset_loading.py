from pathlib import Path

import pytest

from train_face_shape.datasets import FaceShapeFolderDataset


def test_dataset_loader_raises_clear_error_when_folders_missing(tmp_path: Path):
    label_to_index = {"oval": 0, "round": 1}

    with pytest.raises(FileNotFoundError, match="Missing class folders"):
        FaceShapeFolderDataset(tmp_path, label_to_index)
