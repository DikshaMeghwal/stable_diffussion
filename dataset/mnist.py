from typing import Any
from torch.utils.data import Dataset
import glob
import cv2
from numpy.typing import NDArray
import numpy as np


class MNIST(Dataset):
    def __init__(self, data_dir, split, transforms=None) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.image_filenames = glob.glob(f"{self.data_dir}/{self.split}/**/*.png", recursive=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index: Any) -> NDArray:
        image_filename = self.image_filenames[index]
        img = cv2.cvtColor(cv2.imread(image_filename), cv2.COLOR_BGR2GRAY)      # 28 x 28
        img = np.expand_dims(img, axis=2)
        return self.transforms(img) if self.transforms is not None else img