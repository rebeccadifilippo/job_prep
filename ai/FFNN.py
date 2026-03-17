# simple_fcnn.py
# Basic Fully Connected Neural Network in PyTorch

import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== 1. Define the Network =====
class SimpleFCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Input layer (e.g., 4 features)
        self.fc1 = nn.Linear(4, 8)   # 4 → 8

        # Hidden layer
        self.fc2 = nn.Linear(8, 4)   # 8 → 4

        # Output layer
        self.fc3 = nn.Linear(4, 2)   # 4 → 2 (e.g., 2 classes)

    def forward(self, x):
        # x shape: (batch_size, 4)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        # no activation here (handled later if needed)

        return x


# ===== 2. Create Model =====
model = SimpleFCNN()

# ===== 3. Fake Input =====
# batch_size = 1, 4 features
x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

# ===== 4. Forward Pass =====
output = model(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Output:", output)