#ques 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('SLPD.csv')
data = data.drop('Pattern', axis=1, errors='ignore')


# Print column names to diagnose the structure
print("Columns in SLPD.csv:", data.columns.tolist())

# Split the data into training and validation sets (using first 8 rows as training)
train_data = data.iloc[:8]
val_data = data.iloc[8:]

X_train = train_data.iloc[:, :-1].values  
y_train = train_data.iloc[:, -1].values  

X_val = val_data.iloc[:, :-1].values      
y_val = val_data.iloc[:, -1].values     

# Question 2
def step_function(x):
    return np.where(x >= 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)


class Perceptron:
    def __init__(self, input_size, activation_function):
        self.weights = np.zeros(input_size + 1)
        self.activation_function = activation_function

    def predict(self, x):
        x = np.insert(x, 0, 1)
        z = np.dot(self.weights, x)
        output = self.activation_function(z)
        return 1 if output >= 0.5 else 0

#question 3
    def train(self, X, y, epochs=100, lr=0.01):
        for _ in range(epochs):
            for xi, target in zip(X, y):
                xi = np.insert(xi, 0, 1)
                z = np.dot(self.weights, xi)
                output = self.activation_function(z)
                error = target - output
                self.weights += lr * error * xi

#question 4,5,6

def train_and_evaluate(activation_function):
    perceptron = Perceptron(input_size=X_train.shape[1], activation_function=activation_function)
    perceptron.train(X_train, y_train)
    predictions = [perceptron.predict(x) for x in X_val]
    accuracy = np.mean(predictions == y_val)
    return perceptron, accuracy

# Compare decision boundaries
def plot_decision_boundary(perceptron, title):
    # Use only the first two features for visualization
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # For prediction, fill remaining feature values with zeros
    remaining_features = np.zeros((np.ravel(xx).shape[0], max(0, X_train.shape[1] - 2)))
    Z = np.array([perceptron.predict(np.hstack([x1, x2, remaining_features[i]])) 
                 for i, (x1, x2) in enumerate(zip(np.ravel(xx), np.ravel(yy)))])
    
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', marker='o')
    plt.title(title)
    plt.show()

# Question 7
perceptron_step, acc_step = train_and_evaluate(step_function)
plot_decision_boundary(perceptron_step, "Decision Boundary with Step Function")

perceptron_sigmoid, acc_sigmoid = train_and_evaluate(sigmoid)
plot_decision_boundary(perceptron_sigmoid, "Decision Boundary with Sigmoid Function")

perceptron_tanh, acc_tanh = train_and_evaluate(tanh)
plot_decision_boundary(perceptron_tanh, "Decision Boundary with Tanh Function")

perceptron_relu, acc_relu = train_and_evaluate(relu)
plot_decision_boundary(perceptron_relu, "Decision Boundary with ReLU Function")

# Print accuracies
print(f"Accuracy with Step Function: {acc_step}")
print(f"Accuracy with Sigmoid Function: {acc_sigmoid}")
print(f"Accuracy with Tanh Function: {acc_tanh}")
print(f"Accuracy with ReLU Function: {acc_relu}")