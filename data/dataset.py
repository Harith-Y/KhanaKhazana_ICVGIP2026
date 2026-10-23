import csv
import threading
import warnings
from pathlib import Path

from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

_FALLBACK = Image.new("RGB", (224, 224), (0, 0, 0))
_LOAD_TIMEOUT = 8  # seconds before declaring an image hung


def _load_image(path):
    """Load image in a thread with a timeout to avoid PIL infinite hangs."""
    result, exc = [None], [None]

    def _work():
        try:
            result[0] = Image.open(path).convert("RGB")
        except Exception as e:
            exc[0] = e

    t = threading.Thread(target=_work, daemon=True)
    t.start()
    t.join(_LOAD_TIMEOUT)
    if t.is_alive():
        warnings.warn(f"Image load timed out (>{_LOAD_TIMEOUT}s), using blank: {path}")
        return _FALLBACK.copy()
    if exc[0] is not None:
        warnings.warn(f"Unreadable image, using blank: {path}")
        return _FALLBACK.copy()
    return result[0]


class FoodDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.transform = transform
        self.samples = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                p = Path(row["path"])
                try:
                    exists = p.exists()
                except OSError:
                    exists = False
                if exists:
                    self.samples.append((str(p), int(row["label"])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            path, label = self.samples[idx]
        except (ValueError, TypeError):
            warnings.warn(f"Bad sample at idx {idx}, using blank")
            img = self.transform(_FALLBACK.copy()) if self.transform else _FALLBACK.copy()
            return img, 0
        img = _load_image(path)
        if self.transform:
            try:
                img = self.transform(img)
            except Exception:
                warnings.warn(f"Transform failed, using blank: {path}")
                img = self.transform(_FALLBACK.copy())
        return img, label
