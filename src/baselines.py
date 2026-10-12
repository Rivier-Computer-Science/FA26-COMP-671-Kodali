"""Baseline 1 (non-deep): HSV colour histograms + logistic regression / random forest."""
import argparse, json, numpy as np, pandas as pd
from PIL import Image
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from src.utils import load_config, metrics, set_seed

def feat(path, bins=16):
    hsv = np.asarray(Image.open(path).convert("RGB").resize((128, 128)).convert("HSV"))
    return np.concatenate([np.histogram(hsv[..., c], bins=bins, range=(0, 255), density=True)[0] for c in range(3)])

def main(cfg_path):
    cfg = load_config(cfg_path); set_seed(cfg["seed"])
    df = pd.read_csv(cfg["splits_csv"]); X, y = {}, {}
    for s in ["train", "val", "test"]:
        d = df[df.split == s]
        X[s] = np.array(Parallel(n_jobs=-1)(delayed(feat)(p) for p in d.path)); y[s] = d.y.values
    models = {"logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
              "random_forest": RandomForestClassifier(300, n_jobs=-1, random_state=cfg["seed"])}
    out = {}
    for name, m in models.items():
        m.fit(X["train"], y["train"]); pt = m.predict(X["test"])
        out[name] = {"val": metrics(y["val"], m.predict(X["val"])), "test": metrics(y["test"], pt),
                     "test_pred": pt.tolist()}
        print(name, out[name]["val"], out[name]["test"])
    json.dump(out, open("results/baseline_nondeep.json", "w"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/default.yaml")
    main(ap.parse_args().config)
