# FA26-COMP-671-Kodali
# Plant Disease Classification from Leaf Images
**COMP 671 Deep Learning Final Project** — comparing non-deep baselines, an MLP, a plain CNN, and an equal-depth ResNet on PlantVillage.

## Research questions
1. Do skip connections improve generalization vs. an identical-depth plain CNN? (`cnn` vs `resnet`, same width/depth/params ± shortcuts)
2. Does data augmentation reduce overfitting, especially on visually similar disease classes? (`--augment` on/off)

Primary metric: **macro-F1** (test). Secondary: top-1 accuracy. Results are reported as mean ± std over 3 seeds.

## Data
Kaggle "New Plant Diseases Dataset (Augmented)": https://www.kaggle.com/vipoooool/new-plant-diseases-dataset
```bash
kaggle datasets download -d vipoooool/new-plant-diseases-dataset -p data/ --unzip
# put/merge the class folders (train + valid) into data/raw/<class_name>/*.jpg
```
**Split protocol:** images from `train` and `valid` folders are pooled, grouped by source-leaf key (filename stem) and split with `StratifiedGroupKFold` into 70/15/15 (seeded), to limit leakage from augmented copies of the same leaf. `src/check_leakage.py` reports remaining near-duplicates (perceptual hash). Validation is used for early stopping/tuning; the test set is evaluated once at the end.

## Setup
```bash
conda env create -f environment.yml && conda activate plantdl     # or: pip install -r requirements.txt
```
## Run
```bash
python -m src.build_splits                       # writes data/splits.csv, data/classes.csv
python -m src.baselines                          # HSV histogram + LogReg / Random Forest
python -m src.train --model resnet --augment     # models: mlp | cnn | resnet
python -m src.evaluate --model resnet --augment  # test metrics, per-class F1, confusion matrix
python -m src.predict --model resnet --augment --image path/to/leaf.jpg
bash scripts/run_all.sh                          # everything, 3 seeds
```
## Structure
```
configs/default.yaml   hyperparameters      src/models.py     MLP / CNN / ResNet
src/build_splits.py    grouped split        src/train.py      training + early stopping
src/check_leakage.py   pHash leakage check  src/evaluate.py   test eval + confusion
src/baselines.py       non-deep baseline    src/predict.py    single-image inference
scripts/run_all.sh     full experiment      results/ models/  outputs (weights)
```
## Key hyperparameters
See `configs/default.yaml`: AdamW lr 1e-3, wd 1e-4, cosine schedule, batch 64, 30 epochs, patience 5, 128×128 input (32×32 for MLP).

## Ethics & limitations
Not a substitute for expert diagnosis. PlantVillage images are lab-controlled (uniform background); expect a domain gap on field photos. See report.

License: AGPL-3.0 (see LICENSE).
>>>>>>> f3abee8 (Add project files)
