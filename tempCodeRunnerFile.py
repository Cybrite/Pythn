import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.square(x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def load_data():
    # Generate sample data
    X = np.random.randn(100, 2)
    y = np.zeros((100, 2))
    y[X[:, 0] + X[:, 1] > 0, 0] = 1
    y[X[:, 0] + X[:, 1] <= 0, 1] = 1
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

class MLP:
    def __init__(self, input_size, hidden_size, output_size, hidden_activation='relu', output_activation='sigmoid'):
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.01
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.01
        self.bias2 = np.zeros((1, output_size))
        
        # Set hidden layer activation
        if hidden_activation == 'sigmoid':
            self.hidden_activation = sigmoid
            self.hidden_activation_derivative = sigmoid_derivative
        elif hidden_activation == 'relu':
            self.hidden_activation = relu
            self.hidden_activation_derivative = relu_derivative
        elif hidden_activation == 'tanh':
            self.hidden_activation = tanh
            self.hidden_activation_derivative = tanh_derivative

        # Set output layer activation
        if output_activation == 'sigmoid':
            self.output_activation = sigmoid
        elif output_activation == 'relu':
            self.output_activation = relu
        elif output_activation == 'softmax':
            self.output_activation = softmax

    def forward(self, X):
        self.hidden = self.hidden_activation(np.dot(X, self.weights1) + self.bias1)
        self.output = self.output_activation(np.dot(self.hidden, self.weights2) + self.bias2)
        return self.output

    def backward(self, X, y, learning_rate=0.01):
        m = X.shape[0]
        
        # Output layer
        d_output = self.output - y
        d_weights2 = np.dot(self.hidden.T, d_output)
        d_bias2 = np.sum(d_output, axis=0, keepdims=True)
        
        # Hidden layer
        d_hidden = np.dot(d_output, self.weights2.T) * self.hidden_activation_derivative(self.hidden)
        d_weights1 = np.dot(X.T, d_hidden)
        d_bias1 = np.sum(d_hidden, axis=0, keepdims=True)
        
        # Update weights and biases
        self.weights2 -= learning_rate * d_weights2
        self.bias2 -= learning_rate * d_bias2
        self.weights1 -= learning_rate * d_weights1
        self.bias1 -= learning_rate * d_bias1

def compare_activations_and_plot_boundaries():
    X_train, X_test, y_train, y_test = load_data()
    
    # Configurations to test
    configs = [
        {'hidden': 'sigmoid', 'output': 'sigmoid', 'name': 'Sigmoid-Sigmoid'},
        {'hidden': 'relu', 'output': 'sigmoid', 'name': 'ReLU-Sigmoid'},
        {'hidden': 'relu', 'output': 'relu', 'name': 'ReLU-ReLU'},
        {'hidden': 'tanh', 'output': 'sigmoid', 'name': 'Tanh-Sigmoid'},
        {'hidden': 'relu', 'output': 'softmax', 'name': 'ReLU-Softmax'}
    ]
    
    losses = {}
    accuracies = {}
    
    plt.figure(figsize=(15, 5))
    
    for config in configs:
        print(f"\nTraining with {config['name']}:")
        mlp = MLP(X_train.shape[1], 64, y_train.shape[1], 
                 hidden_activation=config['hidden'],
                 output_activation=config['output'])
        
        # Train and record losses
        history = []
        for epoch in range(1000):
            output = mlp.forward(X_train)
            if config['output'] == 'softmax':
                loss = -np.mean(np.sum(y_train * np.log(output + 1e-8), axis=1))
            else:
                loss = -np.mean(y_train * np.log(output + 1e-8) + 
                              (1 - y_train) * np.log(1 - output + 1e-8))
            history.append(loss)
            mlp.backward(X_train, y_train, learning_rate=0.01)
            
            if epoch % 100 == 0:
                # Calculate accuracy
                predictions = np.argmax(mlp.forward(X_test), axis=1)
                true_labels = np.argmax(y_test, axis=1)
                acc = accuracy_score(true_labels, predictions)
                print(f'Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.4f}')
        
        losses[config['name']] = history
        
    # Plot loss convergence
    plt.subplot(1, 2, 1)
    for name, loss in losses.items():
        plt.plot(loss, label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Convergence for Different Configurations')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare_activations_and_plot_boundaries()
