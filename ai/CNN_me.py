# conv2d_forward_practice.py
# Practice file: Implement 2D convolution forward pass from scratch
# Fill in the function marked with TODO. Run the tests to check your solution.

import numpy as np

def conv2d_forward(input, kernel, stride=1, padding=0):
    """
    Implement the forward pass of a 2D convolution.

    Args:
        input: 2D numpy array, shape (H_in, W_in)
        kernel: 2D numpy array, shape (H_k, W_k)
        stride: int
        padding: int

    Returns:
        output: 2D numpy array, shape (H_out, W_out)
    """

    if padding != 0:
        input = np.pad(input, pad_width=padding, mode='constant', constant_values=0)
    
    print(input.shape)

    #output_height = int(((input.shape[0] - kernel.shape[0] + 2*padding)/stride) + 1)
    #output_width = int(((input.shape[1] - kernel.shape[1] + 2*padding)/stride) + 1)
    output_height = int(((input.shape[0] - kernel.shape[0])/stride) + 1)
    output_width = int(((input.shape[1] - kernel.shape[1])/stride) + 1)

    output_matrix = np.zeros((output_height, output_width))

    for h in range(output_height):
        for w in range(output_width):
            h_start = h * stride
            w_start = w  * stride
            h_end = kernel.shape[0] + h_start
            w_end = kernel.shape[1] + w_start
            current_batch = input[h_start:h_end, w_start:w_end]
            output_matrix[h,w]= np.sum(current_batch * kernel)
    return output_matrix

# ===== TEST CASES =====

def test_conv2d_forward():
    # Test 1: Simple 5x5 input with 3x3 kernel
    input1 = np.array([
        [3, 3, 2, 1, 0],
        [0, 0, 1, 3, 1],
        [3, 1, 2, 2, 3],
        [2, 0, 0, 2, 2],
        [2, 0, 0, 0, 1]
    ])
    kernel1 = np.array([
        [0, 1, 2],
        [2, 2, 0],
        [0, 1, 2]
    ])
    output1 = conv2d_forward(input1, kernel1, stride=1, padding=0)
    print("Test 1 output shape (expected (3,3)):", output1.shape if output1 is not None else "Not implemented")
    print(output1)

    # Test 2: Padding test
    output2 = conv2d_forward(input1, kernel1, stride=1, padding=1)
    print("Test 2 output shape (expected (5,5)):", output2.shape if output2 is not None else "Not implemented")
    print(output2)

    # Test 3: Stride test
    output3 = conv2d_forward(input1, kernel1, stride=2, padding=0)
    print("Test 3 output shape (expected (2,2)):", output3.shape if output3 is not None else "Not implemented")
    print(output3)


if __name__ == "__main__":
    test_conv2d_forward()
