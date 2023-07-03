
import torch
print(torch.cuda.is_available())
print("Done")

import pandas as pd

data = pd.read_csv("F://projectData//training_data.csv")
print(data.shape)

data = pd.read_csv("F://projectData//test_data.csv")
print(data.shape)