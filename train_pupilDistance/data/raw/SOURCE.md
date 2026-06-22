# Dataset Source

Requested source:

https://universe.roboflow.com/pranav-p/pupillary-distance-62qse/dataset/1

Use the project importer after Kaggle credentials are configured:

```bash
python eyewear_recommendation_system/.kaggle/import_dataset.py pupil-distance --prepare
```

This dataset contains YOLO annotations for two pupil boxes per face image. The preparation script converts the normalized distance between those box centers into the `pupil_distance` regression target.
