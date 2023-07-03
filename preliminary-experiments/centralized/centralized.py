import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import EMNIST
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

# Define the hyperparameter search space
num_epochs_list = [5, 10, 20]
learning_rates = [0.01, 0.001, 0.0001]
momentum_list = [0.9, 0.95, 0.99]
batch_sizes = [128, 64, 32, 16]

best_accuracy = 0.0
best_hyperparameters = {}


# 1. Dataset Preparation
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = EMNIST(root='./data', split='byclass', train=True, download=True, transform=transform)
test_dataset = EMNIST(root='./data', split='byclass', train=False, download=True, transform=transform)

np.random.seed(42)
torch.manual_seed(42)

# 2. Data Loading
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 3. Model Definition
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # Define your network architecture here

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

model = NeuralNetwork()

# 4. Fine-tune Loop
counter = 0;
for num_epochs in num_epochs_list:
    for lr in learning_rates:
        for momentum in momentum_list:
            for batch_size in batch_sizes:
                accuracy = 0
                print(counter)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=batch_size)
                
                model = NeuralNetwork()
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
                
                for epoch in range(num_epochs):
                    print("****")
                    for images, labels in train_loader:
                        optimizer.zero_grad()
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                model.eval()
                total = 0
                correct = 0

                with torch.no_grad():
                    for images, labels in test_loader:
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                accuracy = correct / total

                print({
                        'accuracy' : accuracy,
                        'num_epochs': num_epochs,
                        'lr': lr,
                        'momentum': momentum,
                        'batch_size': batch_size
                    })

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_hyperparameters = {
                        'num_epochs': num_epochs,
                        'lr': lr,
                        'momentum': momentum,
                        'batch_size': batch_size
                    }
            

print("Best Hyperparameters:")
print(best_hyperparameters)
print("Best Accuracy:", best_accuracy)
