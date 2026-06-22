# Dataset Source

Requested source:

https://www.kaggle.com/datasets/niten19/face-shape-dataset

Use the project importer after Kaggle credentials are configured:

```bash
python eyewear_recommendation_system/.kaggle/import_dataset.py face-shape --prepare
```

The face-shape trainer expects image-classification split folders. If the Kaggle archive uses a different folder name for train/test, the importer keeps the raw download here and `scripts/prepare_dataset.py` can be pointed at the normalized split folders.
