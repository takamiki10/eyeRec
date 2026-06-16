# Raw Dataset Placement

Manually download and extract the Roboflow eye color dataset into this folder.

Expected first-pass format:

```text
data/raw/
├── train/
│   ├── brown/
│   ├── black/
│   ├── blue/
│   ├── green/
│   ├── hazel/
│   └── grey/
├── valid/
│   └── ...
└── test/
    └── ...
```

Then prepare the dataset:

```powershell
python scripts/prepare_dataset.py --raw_dir data/raw --output_dir data/processed
```

If this Roboflow export contains `amber`, the preparation script maps those images into the configured `hazel` class.
