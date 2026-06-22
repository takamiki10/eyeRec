from pathlib import Path

import pytest

from train_eye_shape.datasets import EyeShapeFolderDataset


def test_dataset_loader_raises_clear_error_when_folders_missing(tmp_path: Path):
    label_to_index = {"almond": 0, "round": 1}

    with pytest.raises(FileNotFoundError, match="Missing class folders"):
        EyeShapeFolderDataset(tmp_path, label_to_index)
