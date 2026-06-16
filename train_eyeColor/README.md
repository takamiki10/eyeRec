# Eye Color Training Workspace

## 1. Purpose

This folder trains a local eye color classifier for the EyeRec eyewear recommendation system.

The trained classifier is intended to later replace the placeholder `eye_color_node` in the main application.

This training project is local-only:

- It does not use Roboflow hosted inference.
- It does not require Roboflow API keys.
- It does not download models automatically.
- It assumes the dataset is manually downloaded and placed locally.

## 2. Folder relationship

```text
eyeRec/
├── eyewear_recommendation_system/
└── train_eyeColor/
```

The existing application remains in `eyewear_recommendation_system/`. This training workspace is a sibling folder.

## 3. Create environment

Run these commands in Windows PowerShell:

```powershell
cd C:\Users\tgm10\OneDrive\Documents\eyeRec\train_eyeColor
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## 4. Dataset placement

Manually download the Roboflow eye color dataset from:

https://universe.roboflow.com/fashion-by4cb/eye-color-bc6ji/dataset/4/download

Place the downloaded and extracted dataset under:

```text
data/raw/
```

For the first scaffold, the expected dataset format is image-classification folders:

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

Then run:

```powershell
python scripts/prepare_dataset.py --raw_dir data/raw --output_dir data/processed
```

The script copies `valid` to `val` so training uses:

```text
data/processed/train/
data/processed/val/
data/processed/test/
```

Flat image folders with annotation CSV files may be added later, but this scaffold prioritizes image-classification folder format.

The current Roboflow Folder Structure export may include `amber` instead of `black`. The preparation script merges `amber` images into the configured `hazel` class and creates empty folders for configured labels that are missing from the download, such as `black`.

## 5. Training command

```powershell
python scripts/train_eye_color.py --config configs/train_config.yaml
```

Training saves:

```text
artifacts/checkpoints/
artifacts/exported/eye_color_model.pt
artifacts/exported/label_mapping.json
artifacts/metrics/training_metrics.json
```

## 6. Evaluation command

```powershell
python scripts/evaluate_eye_color.py --config configs/train_config.yaml
```

Evaluation loads `artifacts/exported/eye_color_model.pt`, evaluates on `data/processed/test/`, prints accuracy and per-class accuracy, and saves:

```text
artifacts/metrics/evaluation_metrics.json
```

## 7. Prediction command

```powershell
python scripts/predict_eye_color.py --image data/sample_images/example.jpg
```

Prediction prints:

- `predicted_label`
- `confidence`
- `class_probabilities`

## 8. Export command

```powershell
python scripts/export_to_app.py
```

## 9. Expected exported files

```text
../eyewear_recommendation_system/models/feature_extractors/eye_color/eye_color_model.pt
../eyewear_recommendation_system/models/feature_extractors/eye_color/label_mapping.json
```

The export script creates the destination folder if needed. It only copies model artifacts into the expected app location; it does not modify main application code.

## 10. Note

This training project is local-only and does not use Roboflow hosted inference.

The Roboflow dataset uses the label `grey`. The main application may later map `grey` to `gray` if needed.

Reference only:

- https://github.com/Avaneesh-Pathak/Eye_Color_Detection


## 11. How to Use
```powershell

python scripts/train_eye_color.py --config configs/train_config.yaml
python scripts/evaluate_eye_color.py --config configs/train_config.yaml
```
