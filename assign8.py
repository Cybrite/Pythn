# # #question  2
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def sigmoid_derivative(x):
#     return x * (1 - x)

# def relu(x):
#     return np.maximum(0, x)

# def relu_derivative(x):
#     return np.where(x > 0, 1, 0)

# def tanh(x):
#     return np.tanh(x)

# def tanh_derivative(x):
#     return 1 - np.square(x)

# def softmax(x):
#     exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
#     return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# def load_data():
#     X = np.random.randn(100, 2)
#     y = np.zeros((100, 2))
#     y[X[:, 0] + X[:, 1] > 0, 0] = 1
#     y[X[:, 0] + X[:, 1] <= 0, 1] = 1
    
#     return train_test_split(X, y, test_size=0.2, random_state=42)

# class MLP:
#     def __init__(self, input_size, hidden_size, output_size, hidden_activation='relu', output_activation='sigmoid'):
#         self.weights1 = np.random.randn(input_size, hidden_size) * 0.01
#         self.bias1 = np.zeros((1, hidden_size))
#         self.weights2 = np.random.randn(hidden_size, output_size) * 0.01
#         self.bias2 = np.zeros((1, output_size))
        
#         # Set hidden layer activation
#         if hidden_activation == 'sigmoid':
#             self.hidden_activation = sigmoid
#             self.hidden_activation_derivative = sigmoid_derivative
#         elif hidden_activation == 'relu':
#             self.hidden_activation = relu
#             self.hidden_activation_derivative = relu_derivative
#         elif hidden_activation == 'tanh':
#             self.hidden_activation = tanh
#             self.hidden_activation_derivative = tanh_derivative

#         # Set output layer activation
#         if output_activation == 'sigmoid':
#             self.output_activation = sigmoid
#         elif output_activation == 'relu':
#             self.output_activation = relu
#         elif output_activation == 'softmax':
#             self.output_activation = softmax

#     def forward(self, X):
#         self.hidden = self.hidden_activation(np.dot(X, self.weights1) + self.bias1)
#         self.output = self.output_activation(np.dot(self.hidden, self.weights2) + self.bias2)
#         return self.output

#     def backward(self, X, y, learning_rate=0.01):
#         m = X.shape[0]
        
#         # Output layer
#         d_output = self.output - y
#         d_weights2 = np.dot(self.hidden.T, d_output)
#         d_bias2 = np.sum(d_output, axis=0, keepdims=True)
        
#         # Hidden layer
#         d_hidden = np.dot(d_output, self.weights2.T) * self.hidden_activation_derivative(self.hidden)
#         d_weights1 = np.dot(X.T, d_hidden)
#         d_bias1 = np.sum(d_hidden, axis=0, keepdims=True)
        
#         # Update weights and biases
#         self.weights2 -= learning_rate * d_weights2
#         self.bias2 -= learning_rate * d_bias2
#         self.weights1 -= learning_rate * d_weights1
#         self.bias1 -= learning_rate * d_bias1

# def compare_activations_and_plot_boundaries():
#     X_train, X_test, y_train, y_test = load_data()
    
#     # Configurations to test
#     configs = [
#         # {'hidden': 'sigmoid', 'output': 'sigmoid', 'name': 'Sigmoid-Sigmoid'},
#         {'hidden': 'relu', 'output': 'sigmoid', 'name': 'ReLU-Sigmoid'},
#         # {'hidden': 'relu', 'output': 'relu', 'name': 'ReLU-ReLU'},
#         # {'hidden': 'tanh', 'output': 'sigmoid', 'name': 'Tanh-Sigmoid'},
#         # {'hidden': 'relu', 'output': 'softmax', 'name': 'ReLU-Softmax'}
#     ]
    
#     losses = {}
#     accuracies = {}
    
#     plt.figure(figsize=(15, 5))
    
#     for config in configs:
#         print(f"\nTraining with {config['name']}:")
#         mlp = MLP(X_train.shape[1], 64, y_train.shape[1], 
#                  hidden_activation=config['hidden'],
#                  output_activation=config['output'])
        
#         # Train and record losses
#         history = []
#         for epoch in range(1000):
#             output = mlp.forward(X_train)
#             if config['output'] == 'softmax':
#                 loss = -np.mean(np.sum(y_train * np.log(output + 1e-8), axis=1))
#             else:
#                 loss = -np.mean(y_train * np.log(output + 1e-8) + 
#                               (1 - y_train) * np.log(1 - output + 1e-8))
#             history.append(loss)
#             mlp.backward(X_train, y_train, learning_rate=0.01)
            
#             if epoch % 100 == 0:
#                 # Calculate accuracy
#                 predictions = np.argmax(mlp.forward(X_test), axis=1)
#                 true_labels = np.argmax(y_test, axis=1)
#                 acc = accuracy_score(true_labels, predictions)
#                 print(f'Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.4f}')
        
#         losses[config['name']] = history
        
#     # Plot loss convergence
#     plt.subplot(1, 2, 1)
#     for name, loss in losses.items():
#         plt.plot(loss, label=name)
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Loss Convergence for Different Configurations')
#     plt.legend()
#     plt.grid(True)

#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     compare_activations_and_plot_boundaries()


#question 3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the dataset
df = pd.read_csv("derm.csv")

# Fill missing values in 'age' with the mean
df['age'].fillna(df['age'].mean(), inplace=True)

# Split features and target
X = df.drop(columns=['class']).values
y = df['class'].values

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reduce to 2D using PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Define activation functions and their derivatives
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Define a simple MLP class
class MLP:
    def __init__(self, input_size, hidden_size, output_size, activation='tanh', learning_rate=0.01, epochs=1000):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

        # Set activation function
        if activation == 'tanh':
            self.activation = tanh
            self.activation_derivative = tanh_derivative
        elif activation == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        elif activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        else:
            raise ValueError("Unsupported activation function")

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.activation(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = np.exp(self.z2) / np.sum(np.exp(self.z2), axis=1, keepdims=True)  # Softmax

    def backward(self, X, y):
        m = X.shape[0]
        y_one_hot = np.zeros((m, self.output_size))
        y_one_hot[np.arange(m), y - 1] = 1  # Convert to one-hot encoding

        # Compute gradients
        dz2 = self.a2 - y_one_hot
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.activation_derivative(self.z1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Update weights and biases
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def train(self, X, y):
        for _ in range(self.epochs):
            self.forward(X)
            self.backward(X, y)

    def predict(self, X):
        self.forward(X)
        return np.argmax(self.a2, axis=1) + 1  # Convert back to class labels

# Train MLP with Tanh
mlp_tanh = MLP(input_size=2, hidden_size=10, output_size=len(np.unique(y)), activation='tanh', epochs=1000)
mlp_tanh.train(X_train, y_train)

# Train MLP with ReLU
mlp_relu = MLP(input_size=2, hidden_size=10, output_size=len(np.unique(y)), activation='relu', epochs=1000)
mlp_relu.train(X_train, y_train)

# Train MLP with Sigmoid
mlp_sigmoid = MLP(input_size=2, hidden_size=10, output_size=len(np.unique(y)), activation='sigmoid', epochs=1000)
mlp_sigmoid.train(X_train, y_train)

# Predictions
y_pred_tanh = mlp_tanh.predict(X_test)
y_pred_relu = mlp_relu.predict(X_test)
y_pred_sigmoid = mlp_sigmoid.predict(X_test)

# Accuracy
accuracy_tanh = np.mean(y_pred_tanh == y_test)
accuracy_relu = np.mean(y_pred_relu == y_test)
accuracy_sigmoid = np.mean(y_pred_sigmoid == y_test)
print(f"Tanh MLP Accuracy: {accuracy_tanh * 100:.2f}%")
print(f"ReLU MLP Accuracy: {accuracy_relu * 100:.2f}%")
print(f"Sigmoid MLP Accuracy: {accuracy_sigmoid * 100:.2f}%")

# Function to plot decision boundary
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.jet)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.jet, edgecolors="k")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(title)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.show()

# Plot decision boundaries
plot_decision_boundary(mlp_tanh, X_train, y_train, "Decision Boundary - MLP with Tanh")
plot_decision_boundary(mlp_relu, X_train, y_train, "Decision Boundary - MLP with ReLU")
plot_decision_boundary(mlp_sigmoid, X_train, y_train, "Decision Boundary - MLP with Sigmoid")


