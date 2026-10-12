"""Final evaluation on the held-out test set (run once per final model).
Saves metrics, per-class F1, confusion matrix, top confused pairs."""
import argparse, json, numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score
from src.utils import load_config, get_device, metrics
from src.data import make_loaders
from src.models import build_model
from src.train import predict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--augment", action="store_true")
    ap.add_argument("--seed", type=int, default=None); ap.add_argument("--config", default="configs/default.yaml")
    a = ap.parse_args(); cfg = load_config(a.config)
    seed = a.seed if a.seed is not None else cfg["seed"]
    tag = f"{a.model}{'_aug' if a.augment else ''}_s{seed}"
    size = cfg["mlp_img_size"] if a.model == "mlp" else cfg["img_size"]
    loaders, ncls = make_loaders(cfg, size, False); device = get_device()
    model = build_model(a.model, ncls, cfg).to(device)
    model.load_state_dict(torch.load(f"models/{tag}.pt", map_location=device))
    y, p = predict(model, loaders["test"], device)
    classes = pd.read_csv("data/classes.csv", header=None)[0].tolist()
    cm = confusion_matrix(y, p, labels=range(ncls))
    pc = f1_score(y, p, average=None, labels=range(ncls))
    off = cm.copy(); np.fill_diagonal(off, 0)
    pairs = sorted([(int(off[i, j]), classes[i], classes[j]) for i in range(ncls) for j in range(ncls) if off[i, j]], reverse=True)[:15]
    res = {**metrics(y, p), "per_class_f1": dict(zip(classes, pc.tolist())),
           "top_confusions_true_to_pred": pairs, "y_true": y, "y_pred": p}
    json.dump(res, open(f"results/{tag}_test.json", "w"), indent=2)
    plt.figure(figsize=(14, 12)); sns.heatmap(cm / cm.sum(1, keepdims=True).clip(1), cmap="Blues",
        xticklabels=False, yticklabels=False); plt.title(f"Normalised confusion matrix: {tag}")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.savefig(f"results/{tag}_cm.png", dpi=150, bbox_inches="tight")
    print(metrics(y, p)); print("Top confusions:", pairs[:10])

if __name__ == "__main__":
    main()
