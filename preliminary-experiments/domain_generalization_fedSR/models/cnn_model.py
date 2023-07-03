import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as distributions
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # 5x5 CNN with input channels=1 and output channels=32
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1)
        # 2x2 max pooling layer
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 5x5 CNN with input channels=32 and output channels=64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1)
        # 2x2 max pooling layer
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layer with 2048 neurons
        self.fc1 = nn.Linear(64 * 4 * 4, 2048)
        # Fully connected layer with 62 neurons
        self.fc2 = nn.Linear(2048, 62)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = F.relu(x)

        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def featurize(self, x, num_samples=1):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = F.relu(x)

        x = x.view(x.size(0), -1)

        z_params = self.fc1(x)

        z_mu = z_params[:, :1024]
        z_sigma = F.softplus(z_params[:, 1024:])
        z_dist = distributions.Independent(distributions.normal.Normal(z_mu, z_sigma), 1)
        z = z_dist.rsample([num_samples]).view([-1, 1024])

        return z, z_mu, z_sigma