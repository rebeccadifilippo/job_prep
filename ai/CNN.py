import numpy as np

def conv2d(input, kernel, stride=1, padding=0):
    """
    Perform a 2D convolution manually (supports multi-channel input).

    input: numpy array of shape (C_in, H_in, W_in)
    kernel: numpy array of shape (C_in, H_k, W_k)
    stride: int
    padding: int
    returns: output array of shape (H_out, W_out)
    """
    # Add padding
    if padding > 0:
        input_padded = np.pad(input, ((0,0), (padding, padding), (padding, padding)), mode='constant')
    else:
        input_padded = input

    C_in, H_in, W_in = input_padded.shape
    C_k, H_k, W_k = kernel.shape

    # Output dimensions
    H_out = (H_in - H_k)//stride + 1
    W_out = (W_in - W_k)//stride + 1
    output = np.zeros((H_out, W_out))

    # Perform convolution
    for i in range(H_out):
        for j in range(W_out):
            # Slice the input
            h_start = i*stride
            w_start = j*stride
            h_end = h_start + H_k
            w_end = w_start + W_k
            patch = input_padded[:, h_start:h_end, w_start:w_end]
            
            # Elementwise multiplication and sum over channels
            output[i, j] = np.sum(patch * kernel)

    return output

# ===== Example Usage =====
# Single 3x3 filter on 1-channel 5x5 input
input_1ch = np.array([[1, 2, 0, 1, 2],
                      [0, 1, 2, 2, 0],
                      [1, 0, 1, 0, 1],
                      [2, 1, 0, 1, 2],
                      [0, 2, 1, 2, 1]])
input_1ch = input_1ch[np.newaxis, :, :]  # shape (1, 5, 5)

kernel_1ch = np.array([[1, 0, -1],
                       [1, 0, -1],
                       [1, 0, -1]])
kernel_1ch = kernel_1ch[np.newaxis, :, :]  # shape (1, 3, 3)

output_1ch = conv2d(input_1ch, kernel_1ch, stride=1, padding=0)
print("Single-channel output:\n", output_1ch)

# Multi-channel example: 3-channel RGB input
input_rgb = np.random.randn(3, 5, 5)  # C=3, H=5, W=5
kernel_rgb = np.random.randn(3, 3, 3) # one 3x3 filter across 3 channels
output_rgb = conv2d(input_rgb, kernel_rgb, stride=1, padding=1)
print("RGB output:\n", output_rgb)