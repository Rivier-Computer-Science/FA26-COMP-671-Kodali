"""Inference: python -m src.predict --model resnet --augment --image path/to/leaf.jpg"""
import argparse, pandas as pd, torch
from PIL import Image
from src.utils import load_config, get_device
from src.data import get_transforms
from src.models import build_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--augment", action="store_true")
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--image", required=True)
    ap.add_argument("--config", default="configs/default.yaml"); a = ap.parse_args()
    cfg = load_config(a.config); device = get_device()
    classes = pd.read_csv("data/classes.csv", header=None)[0].tolist()
    size = cfg["mlp_img_size"] if a.model == "mlp" else cfg["img_size"]
    model = build_model(a.model, len(classes), cfg).to(device)
    model.load_state_dict(torch.load(f"models/{a.model}{'_aug' if a.augment else ''}_s{a.seed}.pt", map_location=device)); model.eval()
    x = get_transforms(size, False)(Image.open(a.image).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad(): pr = torch.softmax(model(x), 1)[0]
    top = pr.topk(3)
    for c, i in zip(top.values.tolist(), top.indices.tolist()): print(f"{classes[i]}: {c:.3f}")

if __name__ == "__main__":
    main()
