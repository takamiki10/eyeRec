# Raw Dataset Placement

Place pupil-distance regression data here with `train/`, `valid/`, and `test/` image folders.

Provide labels as either root-level files:

```text
data/raw/train_labels.csv
data/raw/valid_labels.csv
data/raw/test_labels.csv
```

or as `labels.csv` inside each split folder.

Each CSV must include `image_path,pupil_distance`, where `image_path` is relative to the split image folder.
