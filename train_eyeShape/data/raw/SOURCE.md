# Dataset Source

Requested source:

https://universe.roboflow.com/eyeshapes/eye-shapes

This Roboflow project is an object-detection dataset. Download version 2 as YOLOv8, then place/extract it here so the folder contains:

```text
data.yaml
train/images/
train/labels/
valid/images/
valid/labels/
test/images/
test/labels/
```

`scripts/prepare_dataset.py` converts YOLO annotations into classification folders for the labels configured in `configs/label_mapping.yaml`. Classes not in the local schema, such as `deep-set` and `protuding`, are ignored with a warning.
