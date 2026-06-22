# Eyewear Recommendation System

End-to-end local eyewear recommendation pipeline using trained EyeRec feature models.

The system takes a face image, extracts four facial/eye features, aggregates them, and returns the top 3 eyewear recommendations from the local catalog.

## What Runs

```text
input face image
-> eye color model
-> eye shape model
-> pupil distance regressor
-> face shape model
-> feature aggregation
-> rule-based eyewear recommender
-> top 3 recommendations
```

Feature extraction is no longer placeholder-only. The app loads the trained model weights from the sibling training folders:

```text
../train_eyeColor/artifacts/exported/eye_color_model.pt
../train_eyeShape/artifacts/exported/eye_shape_model.pt
../train_faceShape/artifacts/exported/face_shape_model.pt
../train_pupilDistance/artifacts/exported/pupil_distance_model.pt
```

## Current Model Results

Latest local evaluation results:

```text
eye color accuracy:       0.8460
eye shape accuracy:       0.8942
face shape accuracy:      0.8374
pupil distance MAE:       0.0038
pupil distance within .02: 1.0000
```

## Project Layout

```text
eyewear_recommendation_system/
├── configs/
│   ├── feature_schema.yaml
│   ├── model_sources.yaml
│   ├── paths.yaml
│   └── recommender_config.yaml
├── data/
│   ├── eyewear_catalog/eyewear_items.csv
│   └── input_faces/
├── outputs/recommendations/
├── scripts/
│   ├── run_pipeline.py
│   └── run_single_node.py
└── src/eyewear_system/
    ├── feature_nodes/
    ├── model_adapters/
    ├── pipeline/
    └── recommender/
```

## Setup

From the repo root:

```bash
cd eyeRec
source .venv/bin/activate
```

If dependencies are missing:

```bash
pip install -r eyewear_recommendation_system/requirements.txt
pip install -e eyewear_recommendation_system
```

## Run The Full Recommendation Pipeline

Put a face image in:

```text
eyewear_recommendation_system/data/input_faces/
```

Then run:

```bash
cd eyewear_recommendation_system
../.venv/bin/python scripts/run_pipeline.py --image data/input_faces/example.jpg
```

Replace `example.jpg` with any face image placed in `data/input_faces/`.

The output is printed and saved to:

```text
outputs/recommendations/top3_recommendations.json
```

## Example Output

The pipeline returns:

```json
{
  "node_outputs": [
    {"feature_name": "eye_color", "value": "brown"},
    {"feature_name": "eye_shape", "value": "monolid"},
    {"feature_name": "pupil_distance", "value": 0.2513},
    {"feature_name": "face_shape", "value": "square"}
  ],
  "recommendations": [
    {
      "rank": 1,
      "frame_shape": "round",
      "frame_color": "dark brown",
      "reason": "round frames balance a square face; dark brown complements brown eyes"
    }
  ]
}
```

## Run A Single Feature Node

```bash
cd eyewear_recommendation_system
../.venv/bin/python scripts/run_single_node.py --node eye_color --image data/input_faces/example.jpg
../.venv/bin/python scripts/run_single_node.py --node eye_shape --image data/input_faces/example.jpg
../.venv/bin/python scripts/run_single_node.py --node pupil_distance --image data/input_faces/example.jpg
../.venv/bin/python scripts/run_single_node.py --node face_shape --image data/input_faces/example.jpg
```

## Model Details

Eye color:

```text
model: efficientnet_b0
input size: 160
labels: brown, blue, green, hazel, grey
app mapping: grey -> gray
inference: OpenCV eye crops, averaged over detected eyes
extra app behavior: brown-eye prior to reduce blue/green reflection false positives
```

Eye shape:

```text
model: efficientnet_b0
input size: 224
labels: almond, round, monolid, hooded, upturned, downturned
inference: multi-context OpenCV eye crops, left-eye orientation normalized
```

Face shape:

```text
model: mobilenet_v3_large
input size: 224
labels: oval, round, square, heart, oblong
inference: OpenCV face crop before classification
```

Pupil distance:

```text
model: mobilenet_v3_small
input size: 160
output: normalized face-width pupil distance
head: scaled sigmoid regressor
```

## Eyewear Catalog

Catalog file:

```text
data/eyewear_catalog/eyewear_items.csv
```

It currently has 30 eyewear options with distinct frame types such as:

```text
round, soft square, oval, rectangle, cat eye, aviator, square,
browline, wayfarer, geometric, hexagon, panto, rimless,
semi rimless, butterfly, shield, wraparound, clubmaster,
d frame, oversized, keyhole bridge, double bridge, navigator,
rounded square, flat top, transparent, low bridge fit,
thin metal, chunky acetate, sport rectangle
```

The recommender scores catalog items using:

```text
face_shape -> preferred frame types
eye_color -> preferred frame colors
model confidence -> score weighting
```

## Retraining Commands

Run these from each training folder after adding/preparing data.

Eye color:

```bash
cd train_eyeColor
python scripts/prepare_dataset.py --raw_dir data/raw --output_dir data/processed
python scripts/train_eye_color.py --config configs/train_config.yaml
python scripts/evaluate_eye_color.py --config configs/train_config.yaml
```

Eye shape:

```bash
cd train_eyeShape
python scripts/prepare_dataset.py --raw_dir data/raw --output_dir data/processed
python scripts/train_eye_shape.py --config configs/train_config.yaml
python scripts/evaluate_eye_shape.py --config configs/train_config.yaml
```

Face shape:

```bash
cd train_faceShape
python scripts/prepare_dataset.py --raw_dir data/raw --output_dir data/processed
python scripts/train_face_shape.py --config configs/train_config.yaml
python scripts/evaluate_face_shape.py --config configs/train_config.yaml
```

Pupil distance:

```bash
cd train_pupilDistance
python scripts/prepare_dataset.py --raw_dir data/raw --output_dir data/processed
python scripts/train_pupil_distance.py --config configs/train_config.yaml
python scripts/evaluate_pupil_distance.py --config configs/train_config.yaml
```

## Notes

- The recommendation model is currently rule-based, not a trained DNN.
- The feature models are trained PyTorch models loaded through `src/eyewear_system/model_adapters/trained_feature_adapter.py`.
- Pupil distance test accuracy is strong, but the test set is small, so add more labeled examples before treating it as production-grade.
- If the model architecture changes during retraining, rerun evaluation before using the exported `.pt` file in the app.
