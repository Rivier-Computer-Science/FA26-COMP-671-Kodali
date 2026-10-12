"""Build a leakage-aware stratified train/val/test split and save it to CSV.

PlantVillage (Kaggle 'New Plant Diseases Dataset (Augmented)') contains offline-augmented
copies of the same leaf. A naive random split would put copies of one leaf in both train
and test. We therefore (1) group files by a source-leaf key and (2) split by group,
stratified by class. check_leakage.py then verifies with perceptual hashes.
"""
import re, argparse, pathlib
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from src.utils import load_config

def group_key(fname: str) -> str:
    # e.g. '0a769a71-...___RS_HL 1817.JPG' -> '0a769a71-...___RS_HL 1817'
    stem = pathlib.Path(fname).stem
    return re.sub(r"^(flipH|flipV|rot\d*)[_ -]*", "", stem, flags=re.I)

def main(cfg_path):
    cfg = load_config(cfg_path)
    root = pathlib.Path(cfg["data_root"])
    rows = []
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for img in cls_dir.rglob("*"):
            if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                rows.append((str(img), cls_dir.name, cls_dir.name + "|" + group_key(img.name)))
    df = pd.DataFrame(rows, columns=["path", "label", "group"])
    classes = sorted(df.label.unique())
    df["y"] = df.label.map({c: i for i, c in enumerate(classes)})
    print(f"{len(df)} images, {len(classes)} classes, {df.group.nunique()} groups")

    f = cfg["split_fractions"]
    k_test = round(1 / f["test"])
    sgkf = StratifiedGroupKFold(n_splits=k_test, shuffle=True, random_state=cfg["seed"])
    trainval_idx, test_idx = next(sgkf.split(df, df.y, df.group))
    tv = df.iloc[trainval_idx]
    k_val = round((f["train"] + f["val"]) / f["val"])
    sgkf2 = StratifiedGroupKFold(n_splits=k_val, shuffle=True, random_state=cfg["seed"])
    _, va_i = next(sgkf2.split(tv, tv.y, tv.group))
    df["split"] = "train"
    df.loc[tv.index[va_i], "split"] = "val"
    df.loc[df.index[test_idx], "split"] = "test"
    df.to_csv(cfg["splits_csv"], index=False)
    print(df.split.value_counts())
    pd.Series(classes).to_csv("data/classes.csv", index=False, header=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/default.yaml")
    main(ap.parse_args().config)
