# pytorch_practice.py
# PyTorch fundamentals for ML interviews

import torch
import torch.nn as nn
import torch.nn.functional as F

print("\n===== 1. Creating Tensors =====")

a = torch.tensor([1,2,3])
b = torch.zeros((3,3))
c = torch.ones((2,4))
d = torch.randn((3,3))

print("Tensor:", a)
print("Zeros:\n", b)
print("Ones:\n", c)
print("Random:\n", d)


print("\n===== 2. Tensor Shapes =====")

x = torch.randn(2,3,4)

print("Shape:", x.shape)
print("Dimensions:", x.dim())
print("Total elements:", x.numel())


print("\n===== 3. Indexing & Slicing =====")

matrix = torch.arange(16).reshape(4,4)

print("Matrix:\n", matrix)

print("First row:", matrix[0])
print("Second column:", matrix[:,1])
print("Top-left 2x2:\n", matrix[:2,:2])


print("\n===== 4. Reshaping =====")

arr = torch.arange(12)

reshaped = arr.reshape(3,4)

print("Original:", arr)
print("Reshaped:\n", reshaped)

flattened = reshaped.flatten()
print("Flattened:", flattened)


print("\n===== 5. Math Operations =====")

a = torch.tensor([1.,2.,3.])
b = torch.tensor([4.,5.,6.])

print("Addition:", a + b)
print("Elementwise multiply:", a * b)
print("Dot product:", torch.dot(a,b))


print("\n===== 6. Broadcasting =====")

matrix = torch.ones((3,3))
vector = torch.tensor([1.,2.,3.])

result = matrix + vector

print("Matrix:\n", matrix)
print("Vector:", vector)
print("Broadcast result:\n", result)


print("\n===== 7. Reductions =====")

data = torch.randn(4,4)

print("Data:\n", data)

print("Sum:", torch.sum(data))
print("Mean:", torch.mean(data))
print("Max:", torch.max(data))
print("Column sums:", torch.sum(data, dim=0))


print("\n===== 8. Autograd (Gradients) =====")

x = torch.tensor(2.0, requires_grad=True)

y = x**2 + 3*x + 1

y.backward()

print("y:", y)
print("dy/dx:", x.grad)


print("\n===== 9. Simple Linear Layer =====")

linear = nn.Linear(4,2)

input_tensor = torch.randn(1,4)

output = linear(input_tensor)

print("Input:", input_tensor)
print("Output:", output)


print("\n===== 10. Simple Forward Pass =====")

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4,8)
        self.fc2 = nn.Linear(8,2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = SimpleNet()

x = torch.randn(1,4)

y = model(x)

print("Network output:", y)


print("\n===== 11. Convolution Example =====")

conv = nn.Conv2d(
    in_channels=3,
    out_channels=6,
    kernel_size=3,
    stride=1,
    padding=1
)

image = torch.randn(1,3,32,32)

out = conv(image)

print("Input shape:", image.shape)
print("Output shape:", out.shape)


print("\n===== DONE =====")