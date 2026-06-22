# Raw Dataset Placement

Place the Roboflow `pupillary distance` YOLO export here.

Then run:

```powershell
python scripts/prepare_dataset.py --raw_dir data/raw --output_dir data/processed
```

The preparation script converts YOLO pupil boxes into `image_path,pupil_distance` CSV labels.
