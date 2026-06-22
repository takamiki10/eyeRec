# Raw Dataset Placement

Place an eye-shape image-classification dataset here with `train/`, `valid/`, and `test/` splits.

Expected class folders: `almond`, `round`, `monolid`, `hooded`, `upturned`, `downturned`.

Then run:

```powershell
python scripts/prepare_dataset.py --raw_dir data/raw --output_dir data/processed
```
