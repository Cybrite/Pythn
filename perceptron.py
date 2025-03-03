# import numpy as np

# class perceptron:
#     def __init__(self, input_size, Learning_rate=0.1, epochs=10):
#         self.weights = np.zeros(input_size + 1)
#         self.Learning_rate = Learning_rate
#         self.epochs = epochs
    
#     def activation(self, x):
#         # step activation function
#         return 1 if x >= 0 else 0
    
#     def predict(self, inputs):
#         # make a prediction based on current weights
#         summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
#         return self.activation(summation)
    
#     def train(self, X, y):
#         # train the perceptron using perceptron learning rule
#         for _ in range(self.epochs):
#             for inputs, label in zip(X, y):
#                 prediction = self.predict(inputs)
#                 error = label - prediction

#                 self.weights[1:] += self.Learning_rate * error * inputs
#                 self.weights[0] += self.Learning_rate * error

# # Example Dataset: AND Gate
# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Inputs
# y = np.array([0, 0, 0, 1])  # Labels (AND output)

# # Train perceptron
# perceptron = perceptron(input_size=2)
# perceptron.train(X, y)

# # Test perceptron
# print("Testing Perceptron on AND gate:")
# for inputs in X:
#     print(f"Input: {inputs}, Predicted Output: {perceptron.predict(inputs)}")

############################################################################################################

import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=10):
        self.weights = np.zeros(input_size + 1)  # +1 for bias
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def activation(self, x):
        """Step activation function"""
        return 1 if x >= 0 else 0
    
    def predict(self, inputs):
        """Make a prediction based on current weights"""
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]  # Bias term
        return self.activation(summation)
    
    def train(self, X, y):
        """Train the perceptron using the perceptron learning rule"""
        for _ in range(self.epochs):
            for inputs, label in zip(X, y):
                prediction = self.predict(inputs)
                error = label - prediction
                
                # Update weights and bias
                self.weights[1:] += self.learning_rate * error * inputs
                self.weights[0] += self.learning_rate * error

def test_perceptron(perceptron, X, gate_name):
    """Test perceptron and print results"""
    print(f"\nTesting Perceptron on {gate_name} gate:")
    for inputs in X:
        print(f"Input: {inputs}, Predicted Output: {perceptron.predict(inputs)}")

# Define input data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([0, 1, 1, 1])  # OR gate outputs
perceptron_or = Perceptron(input_size=2)
perceptron_or.train(X, y_or)
test_perceptron(perceptron_or, X, "OR")
# XOR Gate Training (Will Fail)
y_xor = np.array([0, 1, 1, 0])  # XOR gate outputs
perceptron_xor = Perceptron(input_size=2)
perceptron_xor.train(X, y_xor)
test_perceptron(perceptron_xor, X, "XOR")
