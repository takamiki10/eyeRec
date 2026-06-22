# Pupil Distance Training Workspace

This folder trains a local image-based pupil distance regressor for the EyeRec eyewear recommendation system.

## Setup

```powershell
cd train_pupilDistance
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Dataset

Place the downloaded Roboflow `pupillary distance` YOLO dataset under `data/raw/`.

Prepare it with:

```powershell
python scripts/prepare_dataset.py --raw_dir data/raw --output_dir data/processed
```

This creates split image folders and `*_labels.csv` files in `data/processed/`.

```powershell
python scripts/train_pupil_distance.py --config configs/train_config.yaml
python scripts/evaluate_pupil_distance.py --config configs/train_config.yaml
python scripts/export_to_app.py
```

Training exports `artifacts/exported/pupil_distance_model.pt` and writes metrics to `artifacts/metrics/training_metrics.json`.
