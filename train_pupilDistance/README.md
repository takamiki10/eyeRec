# Pupil Distance Training Workspace

This folder trains a local pupil distance regressor for the EyeRec eyewear recommendation system.

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

Place split images under `data/raw/train/`, `data/raw/valid/`, and `data/raw/test/`.
Add labels either as `data/raw/train_labels.csv`, `data/raw/valid_labels.csv`, `data/raw/test_labels.csv`, or as `labels.csv` inside each split folder.

Each labels CSV must contain:

```csv
image_path,pupil_distance
example.jpg,0.48
```

`image_path` is relative to that split image folder. `pupil_distance` is expected to be normalized by face width.

Prepare it with:

```powershell
python scripts/prepare_dataset.py --raw_dir data/raw --output_dir data/processed
```

## Commands

```powershell
python scripts/train_pupil_distance.py --config configs/train_config.yaml
python scripts/evaluate_pupil_distance.py --config configs/train_config.yaml
python scripts/predict_pupil_distance.py --image data/sample_images/example.jpg
python scripts/export_to_app.py
```

Training exports `artifacts/exported/pupil_distance_model.pt`.
Evaluation reports MAE and RMSE. Prediction prints `pupil_distance` in `normalized_face_width` units.
The app export destination is `../eyewear_recommendation_system/models/feature_extractors/pupil_distance/`.
