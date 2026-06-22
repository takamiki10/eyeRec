# Raw Dataset Placement

Place a face-shape image-classification dataset here with `train/`, `valid/`, and `test/` splits.

Expected class folders: `oval`, `round`, `square`, `heart`, `oblong`.

Then run:

```powershell
python scripts/prepare_dataset.py --raw_dir data/raw --output_dir data/processed
```
