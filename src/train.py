"""Train mlp / cnn / resnet with optional augmentation. Early stopping on val macro-F1."""
import argparse, json, torch, torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from src.utils import load_config, set_seed, get_device, metrics
from src.data import make_loaders
from src.models import build_model, count_params

@torch.no_grad()
def predict(model, loader, device):
    model.eval(); ys, ps = [], []
    for x, y in loader:
        ps += model(x.to(device)).argmax(1).cpu().tolist(); ys += y.tolist()
    return ys, ps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["mlp", "cnn", "resnet"])
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--config", default="configs/default.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    seed = a.seed if a.seed is not None else cfg["seed"]
    set_seed(seed); device = get_device()
    size = cfg["mlp_img_size"] if a.model == "mlp" else cfg["img_size"]
    loaders, ncls = make_loaders(cfg, size, a.augment)
    model = build_model(a.model, ncls, cfg).to(device)
    tag = f"{a.model}{'_aug' if a.augment else ''}_s{seed}"
    print(f"{tag}: {count_params(model):,} params on {device}")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])
    crit = nn.CrossEntropyLoss()
    Path("models").mkdir(exist_ok=True); Path("results").mkdir(exist_ok=True)
    best, bad, hist = -1, 0, []
    for ep in range(cfg["epochs"]):
        model.train(); tl, n = 0, 0
        for x, y in tqdm(loaders["train"], desc=f"ep{ep}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
            tl += loss.item() * len(y); n += len(y)
        sched.step()
        vy, vp = predict(model, loaders["val"], device); m = metrics(vy, vp)
        hist.append({"epoch": ep, "train_loss": tl / n, **{f"val_{k}": v for k, v in m.items()}})
        print(hist[-1])
        if m["macro_f1"] > best:
            best, bad = m["macro_f1"], 0; torch.save(model.state_dict(), f"models/{tag}.pt")
        else:
            bad += 1
            if bad >= cfg["patience"]: print("early stop"); break
    json.dump({"params": count_params(model), "best_val_macro_f1": best, "history": hist},
              open(f"results/{tag}_train.json", "w"), indent=2)

if __name__ == "__main__":
    main()
