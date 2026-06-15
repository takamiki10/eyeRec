# Eyewear Recommendation System

Preliminary Python scaffold for an eyewear recommendation pipeline. The current implementation is intentionally placeholder-based: it does not download models, does not use Roboflow API keys, and does not run real inference.

Recommended Python version: 3.10, 3.11, or 3.12.

## Pipeline Overview

User face image -> preprocessing -> four feature-detection nodes -> feature aggregation -> ANN/DNN recommender placeholder -> top 3 eyewear recommendations.

Feature nodes:

- Eye color detection
- Eye shape detection
- Pupil distance / eye distance estimation
- Face shape detection

## Folder Structure

```text
eyewear_recommendation_system/
├── configs/
├── data/
├── models/
├── outputs/
├── scripts/
├── src/eyewear_system/
└── tests/
```

## Installation

```bash
cd eyewear_recommendation_system
py -3.12 -m venv .venv (Check with py -3.13 --version)
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

On macOS/Linux, activate with `source .venv/bin/activate`.

## Run the Full Pipeline

Place an image at `data/input_faces/example.jpg`, then run:

```bash
python scripts/run_pipeline.py --image data/input_faces/example.jpg
```

The output is written to:

```text
outputs/recommendations/top3_recommendations.json
```

## Run One Feature Node

```bash
python scripts/run_single_node.py --node eye_color --image data/input_faces/example.jpg
python scripts/run_single_node.py --node eye_shape --image data/input_faces/example.jpg
python scripts/run_single_node.py --node pupil_distance --image data/input_faces/example.jpg
python scripts/run_single_node.py --node face_shape --image data/input_faces/example.jpg
```

## Run Tests

```bash
pytest
```

## Model Status

All model inference is currently placeholder-based. The source URLs and local placeholder model paths are documented in `configs/model_sources.yaml` so real adapters and model files can be inserted later.
