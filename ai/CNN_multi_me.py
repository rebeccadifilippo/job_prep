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
    if padding>0:
        #input_padded = np.pad(input,pad_width = padding, mode ='constant',constant_values = '0')
        input_padded = np.pad(input,((0,0),(padding,padding),(padding,padding)),mode='constant')
    else:
        input_padded = input
    
    C_in, H_in ,W_in = input_padded.shape
    C_k , H_k, W_k = kernel.shape

    output_width = int((W_in - W_k )/ stride)
    output_height = int((W_in - H_k) / stride)

    output = np.zeros((output_height,output_width ))

    for h in range(output_height):
        for w in range(output_width):
            h_start = h*stride
            w_start = w*stride
            h_end = h_start + H_k
            w_end = w_start + W_k

            curr_batch = input_padded[:,h_start:h_end,w_start:w_end]
            output[h,w] = np.sum(curr_batch * kernel)
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

output_1ch = conv2d(input_1ch, kernel_1ch, stride=1, padding=1)
print("Single-channel output:\n", output_1ch)

# Multi-channel example: 3-channel RGB input
input_rgb = np.random.randn(3, 5, 5)  # C=3, H=5, W=5
kernel_rgb = np.random.randn(3, 3, 3) # one 3x3 filter across 3 channels
output_rgb = conv2d(input_rgb, kernel_rgb, stride=1, padding=1)
print("RGB output:\n", output_rgb)