import audio
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

class MelDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path) as f:
            self._files = []
            for l in f:
                l = l.strip()
                self._files.append(l)

    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx):
        p = self._files[idx]
        f = np.load(p)
        return f['wav'].astype(np.float32), f['spec'].astype(np.float32)
