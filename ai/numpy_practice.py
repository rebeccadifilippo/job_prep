# numpy_practice.py
# A beginner-to-intermediate NumPy practice script for ML interviews

import numpy as np

print("\n===== 1. Creating Arrays =====")

a = np.array([1, 2, 3])
b = np.zeros((3, 3))
c = np.ones((2, 4))
d = np.random.randn(3, 3)

print("Array:", a)
print("Zeros:\n", b)
print("Ones:\n", c)
print("Random:\n", d)


print("\n===== 2. Array Shapes =====")

x = np.random.randn(2, 3, 4)

print("Shape:", x.shape)
print("Dimensions:", x.ndim)
print("Total elements:", x.size)


print("\n===== 3. Indexing & Slicing =====")

matrix = np.arange(16).reshape(4,4)

print("Matrix:\n", matrix)

print("First row:", matrix[0])
print("Second column:", matrix[:,1])
print("Top-left 2x2:\n", matrix[:2, :2])


print("\n===== 4. Reshaping =====")

arr = np.arange(12)

reshaped = arr.reshape(3,4)

print("Original:", arr)
print("Reshaped 3x4:\n", reshaped)

flattened = reshaped.flatten()
print("Flattened:", flattened)


print("\n===== 5. Basic Math Operations =====")

a = np.array([1,2,3])
b = np.array([4,5,6])

print("Addition:", a + b)
print("Multiplication:", a * b)
print("Dot product:", np.dot(a,b))


print("\n===== 6. Broadcasting =====")

matrix = np.ones((3,3))
vector = np.array([1,2,3])

print("Matrix:\n", matrix)
print("Vector:", vector)

result = matrix + vector
print("Broadcast result:\n", result)


print("\n===== 7. Summaries =====")

data = np.random.randn(4,4)

print("Data:\n", data)

print("Sum:", np.sum(data))
print("Mean:", np.mean(data))
print("Max:", np.max(data))
print("Column sums:", np.sum(data, axis=0))


print("\n===== 8. Padding (important for CNNs) =====")

img = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])

padded = np.pad(img, ((1,1),(1,1)), mode='constant')

print("Original:\n", img)
print("Padded:\n", padded)


print("\n===== 9. Sliding Window (for Convolution) =====")

image = np.arange(25).reshape(5,5)

kernel = np.array([
    [1,0,-1],
    [1,0,-1],
    [1,0,-1]
])

print("Image:\n", image)
print("Kernel:\n", kernel)

output = np.zeros((3,3))

for i in range(3):
    for j in range(3):

        patch = image[i:i+3, j:j+3]

        output[i,j] = np.sum(patch * kernel)

print("Convolution result:\n", output)


print("\n===== 10. Multi-channel Example =====")

rgb_image = np.random.randn(3,5,5)
kernel = np.random.randn(3,3,3)

conv_sum = np.sum(rgb_image[:,0:3,0:3] * kernel)

print("Example multi-channel convolution value:", conv_sum)


print("\n===== DONE =====")