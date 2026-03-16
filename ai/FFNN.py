import numpy as np

class SimpleFFNN:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights with small random values (He initialization style)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return Z > 0

    def softmax(self, Z):
        # Numerically stable softmax (The 'Subtract Max' trick)
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def forward(self, X):
        # Layer 1
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        
        # Layer 2
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2

    def backward(self, X, y_true, learning_rate=0.01):
        m = X.shape[0] # Batch size
        
        # Calculate Output Error (Cross-entropy gradient)
        dZ2 = self.A2 - y_true
        dW2 = (1 / m) * np.dot(self.A1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
        
        # Backprop through Hidden Layer
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)
        
        # Update Weights (Gradient Descent)
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

# --- UNIT TEST ---
def test_ffnn():
    X = np.array([[0.5, 0.1]]) # 1 sample, 2 features
    y = np.array([[1, 0]])     # One-hot encoded label
    
    model = SimpleFFNN(input_size=2, hidden_size=4, output_size=2)
    
    # Check forward pass
    output = model.forward(X)
    print(f"Initial Output Probs: {output}")
    
    # Perform one update
    model.backward(X, y)
    new_output = model.forward(X)
    print(f"Output after 1 step: {new_output}")



test_ffnn()