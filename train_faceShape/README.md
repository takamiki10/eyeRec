# Face Shape Training Workspace

This folder trains a local face shape classifier for the EyeRec eyewear recommendation system.

## Setup

```powershell
cd train_faceShape
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Dataset

Place an image-classification dataset under `data/raw/`:

```text
data/raw/
├── train/
│   ├── oval/
│   ├── round/
│   ├── square/
│   ├── heart/
│   └── oblong/
├── valid/
│   └── ...
└── test/
    └── ...
```

Prepare it with:

```powershell
python scripts/prepare_dataset.py --raw_dir data/raw --output_dir data/processed
```

## Commands

```powershell
python scripts/train_face_shape.py --config configs/train_config.yaml
python scripts/evaluate_face_shape.py --config configs/train_config.yaml
python scripts/predict_face_shape.py --image data/sample_images/example.jpg
python scripts/export_to_app.py
```

Training exports `artifacts/exported/face_shape_model.pt` and `artifacts/exported/label_mapping.json`.
The app export destination is `../eyewear_recommendation_system/models/feature_extractors/face_shape/`.
