import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

def get_transforms(size, augment):
    if augment:
        tf = [T.Resize((size, size)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
              T.RandomRotation(20), T.ColorJitter(0.3, 0.3, 0.3, 0.05)]
    else:
        tf = [T.Resize((size, size))]
    return T.Compose(tf + [T.ToTensor(), T.Normalize(MEAN, STD)])

class LeafDataset(Dataset):
    def __init__(self, df, transform):
        self.paths, self.y, self.tf = df.path.tolist(), df.y.tolist(), transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB")), self.y[i]

def make_loaders(cfg, size, augment):
    df = pd.read_csv(cfg["splits_csv"])
    out = {}
    for s in ["train", "val", "test"]:
        ds = LeafDataset(df[df.split == s], get_transforms(size, augment and s == "train"))
        out[s] = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=(s == "train"),
                            num_workers=cfg["num_workers"], pin_memory=True)
    return out, df.y.nunique()
