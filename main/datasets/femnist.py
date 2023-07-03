import numpy as np
import torch

import datasets.np_transforms as tr

from typing import Any
from torch.utils.data import Dataset

IMAGE_SIZE = 28

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ])


class Femnist(Dataset):

    def __init__(self,
                 data: dict,
                 transform: tr.Compose,
                 client_name: str):
        super().__init__()
        self.samples = [(image, label) for image, label in zip(data['x'], data['y'])]
        self.transform = transform
        self.client_name = client_name

    def __getitem__(self, index: int) -> Any:

        image = self.samples[index][0]

        label = [self.samples[index][1]]
        x, y = torch.Tensor(image).view(1, IMAGE_SIZE, IMAGE_SIZE), torch.Tensor(label)

        ##
        x = self.transform.transforms[1](x)
        ##

        return x, y

    def __len__(self) -> int:
        return len(self.samples)
