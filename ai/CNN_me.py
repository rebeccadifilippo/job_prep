# conv2d_forward_practice.py

import numpy as np
import math

def conv2d_forward(input, kernel, stride=1, padding=0):
    if padding != 0:
        input = np.pad(input, pad_width=padding, mode='constant', constant_values=0)
    

    output_height = int(((input.shape[0] - kernel.shape[0]) / stride) + 1)
    output_width = int(((input.shape[1] - kernel.shape[1]) / stride) + 1)

    output_matrix = np.zeros((output_height, output_width))

    for h in range(output_height):
        for w in range(output_width):
            h_start = h * stride
            w_start = w * stride
            h_end = kernel.shape[0] + h_start
            w_end = kernel.shape[1] + w_start

            current_batch = input[h_start:h_end, w_start:w_end]
            output_matrix[h, w] = np.sum(current_batch * kernel)

    return output_matrix


def maxpool_forward(input, pool_size=2, stride=2):
    H_out = int((input.shape[0] - pool_size) / stride + 1)
    W_out = int((input.shape[1] - pool_size) / stride + 1)

    output = np.zeros((H_out, W_out))

    for h in range(H_out):
        for w in range(W_out):
            h_start = h * stride
            w_start = w * stride
            h_end = h_start + pool_size
            w_end = w_start + pool_size

            current_patch = input[h_start:h_end, w_start:w_end]
            output[h, w] = np.max(current_patch)

    return output


# ===== activation (FILL THIS IN) =====
def activation(x,relu):
    """
    TODO: Implement activation

    Args:
        x: 1D numpy array

    Returns:
        probabilities: same shape as x
    """
    #SOFTMAX
    if relu == 0:
        #Softmax converts these numbers into probabilities that sum to 1
        # #soft max is e^x / sum e^x
        #When values in x are large, np.exp can overflow. The stable version subtracts the max:
        shift_x = x - np.max(x)
        exp_x = np.exp(shift_x)
        return exp_x / np.sum(exp_x)
    #RELU
    elif relu == 1:
        return np.maximum(0, x)



# ===== TEST HELPERS =====
def test_maxpool(input_matrix):
    print("\n--- MaxPool ---")
    print("Input shape:", input_matrix.shape)

    output = maxpool_forward(input_matrix, pool_size=2, stride=2)

    print("Output shape:", output.shape)
    print(output)

    return output


def test_pipeline(conv_output):
    """
    Simulates:
    Conv → Pool → Flatten → activation
    """

    pooled = test_maxpool(conv_output)

    print("\n--- Flatten ---")
    flat = pooled.flatten()  
    print("Flattened shape:", flat.shape)
    print(flat)

    print("\n--- activation ---")
    probs = activation(flat,0)   
    print(probs)


# ===== MAIN TEST =====
def test_conv2d_forward():
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
    print("\n===== INPUT MATRIX =====")
    print(input1)
    print("\n===== KERNEL/FILTER MATRIX =====")
    print(kernel1)

    print("\n===== TEST 1: NO PADDING =====")
    output1 = conv2d_forward(input1, kernel1, stride=1, padding=0)
    print("Output:\n", output1)
    test_pipeline(output1)

    print("\n===== TEST 2: WITH PADDING =====")
    output2 = conv2d_forward(input1, kernel1, stride=1, padding=1)
    print("Output:\n", output2)
    test_pipeline(output2)

    ''' print("\n===== TEST 3: STRIDE 2 =====")
    output3 = conv2d_forward(input1, kernel1, stride=2, padding=0)
    print("Output:\n", output3)
    test_pipeline(output3)'''


if __name__ == "__main__":
    test_conv2d_forward()