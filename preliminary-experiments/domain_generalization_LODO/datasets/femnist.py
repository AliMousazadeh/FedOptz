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

#hossein: I don't know what you have done above
#hossein: here I add new transformation on Femnist Objects,
#remember that dataset within each client is made up of Femnist objects
import torchvision.transforms.functional as F


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
        # TODO: missing code here!

        image = self.samples[index][0]
        label = [self.samples[index][1]]
        x, y = torch.Tensor(image).view(1, IMAGE_SIZE, IMAGE_SIZE), torch.Tensor(label)
        x = self.transform.transforms[1](x)
        return x, y

    def __len__(self) -> int:
        return len(self.samples)

    # I also added this new transformation to the images if the function in called
    def apply_new_transformation(self, rotation_angle: float):

        for idx in range(len(self.samples)):
            image = self.samples[idx][0]
            label = self.samples[idx][1]
            x = torch.Tensor(image).view(1, IMAGE_SIZE, IMAGE_SIZE)
            # Apply new rotation transformation
            x = F.rotate(x, rotation_angle)

            self.samples[idx] = (x, label)

    def get_samples(self):
        return self.samples

"""x.squeeze(): This refers to the transformed image. 
The squeeze() function is used to remove any dimensions with a size of 1.
In this case, it removes the extra dimension added during the transformation process, resulting in a tensor with shape
(IMAGE_SIZE, IMAGE_SIZE) instead of (1, IMAGE_SIZE, IMAGE_SIZE)
By assigning the new tuple (x.squeeze(), label[0]) to self.samples[idx],
the transformed image and its label are updated in the samples list at the specified index. 
This allows you to store the transformed data back into the dataset for future use or access."""