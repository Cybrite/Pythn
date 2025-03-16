import numpy as np
import matplotlib.pyplot as plt
class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, epochs=10000):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        # Initialize weights and biases
        self.W1 = np.random.randn(self.hidden_size, self.input_size)
        self.b1 = np.zeros((self.hidden_size, 1))
        self.W2 = np.random.randn(self.output_size, self.hidden_size)
        self.b2 = np.zeros((self.output_size, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        """Forward propagation"""
        self.Z1 = np.dot(self.W1, X.T) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.A2

    def backward(self, X, y):
        """Backward propagation"""
        m = X.shape[0]
        y = y.reshape(1, m)
        
        # Compute error
        dZ2 = self.A2 - y
        dW2 = (1 / m) * np.dot(dZ2, self.A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
        
        dZ1 = np.dot(self.W2.T, dZ2) * self.sigmoid_derivative(self.A1)
        dW1 = (1 / m) * np.dot(dZ1, X)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
       
        # Update weights and biases
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def train(self, X, y):
        """Train the network"""
        for _ in range(self.epochs):
            self.forward(X)
            self.backward(X, y)

    def predict(self, X):
        """Make predictions"""
        predictions = self.forward(X)
        return (predictions > 0.5).astype(int)

# XOR Dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Train MLP
mlp = MLP(input_size=2, hidden_size=2, output_size=1, learning_rate=0.5, epochs=10000)
mlp.train(X, y)

# Test MLP
print("\nMLP Predictions for XOR:")
for i, inputs in enumerate(X):
    prediction = mlp.predict(inputs.reshape(1, -1))
    print(f"Input: {inputs}, Predicted Output: {prediction[0][0]}")

# Visualization of Decision Boundary
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    # Predict for each point in the meshgrid
    Z = np.array([model.predict(np.array([[a, b]])) for a, b in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap=plt.cm.Paired, marker="o", s=100)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("MLP Decision Boundary for XOR")
    plt.show()

# Plot decision boundary
plot_decision_boundary(mlp, X, y)
