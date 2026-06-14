## Training (local)

This repo ships **pretrained** TensorFlow SavedModels under:
- `potato_trained_models/1`
- `tomato_trained_models/1`
- `pepper_trained_models/1`

To **retrain** and verify accuracy/confidence requirements, you need the dataset locally.

### Dataset

The original notebooks use Kaggle dataset **`arjuntejaswi/plant-village`**, extracted so you have a folder like:

`PlantVillage/<class_name>/*.jpg`

Example class folders used by this app:
- Potato: `Potato___Early_blight`, `Potato___Late_blight`, `Potato___healthy`
- Pepper: `Pepper__bell___Bacterial_spot`, `Pepper__bell___healthy`
- Tomato: `Tomato_healthy`, `Tomato_Spider_mites_Two_spotted_spider_mite`, ...

### Optional: Kaggle download

If you have `kaggle.json`, place it at `~/.kaggle/kaggle.json` and run:

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m train.download_kaggle --out_dir data
```

This creates `data/PlantVillage/`.

### Train + evaluate + calibrate

```bash
.venv/bin/python -m train.train_plant --plant potato --data_dir data/PlantVillage --out_dir potato_trained_models/1
.venv/bin/python -m train.train_plant --plant pepper --data_dir data/PlantVillage --out_dir pepper_trained_models/1
.venv/bin/python -m train.train_plant --plant tomato --data_dir data/PlantVillage --out_dir tomato_trained_models/1
```

Each run prints:
- test top-1 accuracy (must be ≥ 0.90 to meet your requirement)
- temperature scaling value (for confidence calibration)

