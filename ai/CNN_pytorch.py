# simple_cnn.py
# Basic Convolutional Neural Network in PyTorch

import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== 1. Define the Network =====
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Input: (batch_size, 1, 28, 28)  -> grayscale image

        # Convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3)
        # Output: (batch_size, 4, 26, 26)

        # Second conv layer
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3)
        # Output: (batch_size, 8, 24, 24)

        # Fully connected layers
        self.fc1 = nn.Linear(8 * 24 * 24, 16)
        self.fc2 = nn.Linear(16, 2)  # 2 classes

    def forward(self, x):
        # x shape: (batch_size, 1, 28, 28)

        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        # Flatten
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)

        return x


# ===== 2. Create Model =====
model = SimpleCNN()

# ===== 3. Fake Input =====
# batch_size = 1, 1 channel, 28x28 image
x = torch.randn(1, 1, 28, 28)

# ===== 4. Forward Pass =====
output = model(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Output:", output)