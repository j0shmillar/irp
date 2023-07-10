import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List

import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

class MODIS_new(Dataset):
    def __init__(self, in_files: List[str], tar_files: List[str]):
        self.input_files = sorted(in_files)
        self.target_files = sorted(tar_files)

    def __len__(self) -> int:
        return len(self.input_files)

    def __getitem__(self, item: int) -> Tuple[np.ndarray, np.ndarray]:
        input = np.load(self.input_files[item])
        target = np.load(self.target_files[item])

        return input, target
