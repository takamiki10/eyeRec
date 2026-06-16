from pathlib import Path

import pytest

from train_eye_color.datasets import EyeColorFolderDataset


def test_dataset_loader_raises_clear_error_when_folders_missing(tmp_path: Path):
    label_to_index = {"brown": 0, "black": 1}

    with pytest.raises(FileNotFoundError, match="Missing class folders"):
        EyeColorFolderDataset(tmp_path, label_to_index)
