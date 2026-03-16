# tensor_practice_interview.py
# Practice file for tensor manipulation exercises (NumPy)
# Fill in the functions below. Run the test cases to check your solutions.

import numpy as np

# ===== 1. 2D Convolution =====
def conv2d(input, kernel, stride=1, padding=0):
    """
    Implement 2D convolution from scratch.
    
    input: 2D numpy array (H, W) or 3D numpy array (C, H, W)
    kernel: same channels as input (C, H_k, W_k)
    stride: int
    padding: int
    
    Returns: 2D array (H_out, W_out)
    """
    # TODO: Implement convolution
    pass

# Test Conv2D
input_matrix = np.arange(25).reshape(5,5)
kernel = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
conv_output = conv2d(input_matrix, kernel, stride=1, padding=0)
print("Conv2D Output:\n", conv_output)
print("Expected shape: (3,3)")
print("Actual shape:", conv_output.shape if conv_output is not None else "Not implemented")


# ===== 2. Flatten / Reshape Tensor =====
def flatten_tensor(batch_tensor):
    """
    Flatten a tensor of shape (batch_size, channels, height, width)
    to (batch_size, channels*height*width)
    """
    # TODO: Implement flatten
    pass

# Test Flatten
batch = np.random.randn(32, 3, 32, 32)
flat = flatten_tensor(batch)
print("Flattened shape:", flat.shape)
print("Expected shape: (32, 3*32*32)")


# ===== 3. Axis Operations =====
def mean_across_channels(x):
    """
    Compute mean across channels.
    x: shape (batch, channels, height, width)
    Returns shape: (batch, height, width)
    """
    # TODO: Implement mean across channels
    pass

def max_across_height(x):
    """
    Compute max across height axis.
    x: shape (batch, channels, height, width)
    Returns shape: (batch, channels, width)
    """
    # TODO: Implement max across height
    pass

# Test Axis Operations
x = np.random.randn(32, 3, 32, 32)

mean_ch = mean_across_channels(x)
print("Mean across channels shape:", mean_ch.shape)
print("Expected shape: (32, 32, 32)")

max_h = max_across_height(x)
print("Max across height shape:", max_h.shape)
print("Expected shape: (32, 3, 32)")


# ===== 4. Bonus: Multi-channel Convolution =====
def conv2d_multi_channel(input, kernels, stride=1, padding=0):
    """
    Implement multi-channel convolution.
    
    input: (C_in, H, W)
    kernels: (C_out, C_in, H_k, W_k)
    stride: int
    padding: int
    
    Returns: (C_out, H_out, W_out)
    """
    # TODO: Implement multi-channel conv
    pass

# Test multi-channel conv
input_rgb = np.random.randn(3, 5, 5)
kernels_rgb = np.random.randn(2, 3, 3, 3)  # 2 filters
out_multi = conv2d_multi_channel(input_rgb, kernels_rgb, stride=1, padding=1)
print("Multi-channel Conv shape:", out_multi.shape)
print("Expected shape: (2, 5, 5)")