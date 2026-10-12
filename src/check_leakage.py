"""Report near-duplicate images that cross train/test using perceptual hashes (sampled)."""
import argparse, pandas as pd, imagehash
from PIL import Image
from tqdm import tqdm
from src.utils import load_config

def main(cfg_path, n=2000, thresh=2):
    cfg = load_config(cfg_path)
    df = pd.read_csv(cfg["splits_csv"])
    test = df[df.split == "test"].sample(min(n, (df.split == "test").sum()), random_state=0)
    train = df[df.split == "train"].sample(min(n * 3, (df.split == "train").sum()), random_state=0)
    h = lambda p: imagehash.phash(Image.open(p).convert("RGB"))
    th = [h(p) for p in tqdm(train.path, desc="train hashes")]
    hits = sum(any(h(p) - x <= thresh for x in th) for p in tqdm(test.path, desc="test hashes"))
    print(f"Near-duplicate test images (pHash dist<={thresh}) vs train sample: {hits}/{len(test)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/default.yaml")
    main(ap.parse_args().config)
