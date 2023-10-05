# Josh Millar: edsml-jm4622

import torch
import warnings

from typing import Tuple

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


class DataSet(torch.utils.data.Dataset):
    "Custom dataset wrapper."

    def __init__(self, in_path: str, tar_path: str):
        """
        Args:
            in_path (str): Path to the input data, in tensor format.
            tar_path (str): Path to the target data, in tensor format.
        """
        self.input = torch.load(in_path)
        self.target = torch.load(tar_path)

    def __len__(self) -> int:
        """
        Get the number of elements in the dataset.

        Returns:
            int: Number of elements in the dataset.
        """
        return len(self.input)

    def __getitem__(self, item) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a specific element from the dataset.

        Args:
            item (int): Index of the element to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the input and target tensors at specified index (item).
        """
        return self.input[item], self.target[item]
