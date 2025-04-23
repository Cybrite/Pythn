import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
import time

# Load the dataset
data = pd.read_csv('d:/Pythn/HAM10000_metadata.csv')

# Basic data exploration
print("Dataset shape:", data.shape)
print("\nClass distribution:")
print(data['dx'].value_counts())

# Handling missing values
data['age'].fillna(data['age'].median(), inplace=True)
data.loc[data['age'] == 0.0, 'age'] = data['age'].median()
data['sex'].fillna(data['sex'].mode()[0], inplace=True)
data['localization'].fillna(data['localization'].mode()[0], inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
data['dx_encoded'] = label_encoder.fit_transform(data['dx'])
data['sex_encoded'] = label_encoder.fit_transform(data['sex'])
data['localization_encoded'] = label_encoder.fit_transform(data['localization'])

# One-hot encode the target variable
onehot_encoder = OneHotEncoder(sparse_output=False)
dx_onehot = onehot_encoder.fit_transform(data['dx_encoded'].values.reshape(-1, 1))
num_classes = dx_onehot.shape[1]
class_names = label_encoder.classes_

# Define features and target
X = data[['age', 'sex_encoded', 'localization_encoded']].values
y = dx_onehot

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class MLP:
    def __init__(self, input_size, hidden_size, output_size, activation='relu'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        
        # Initialize weights and biases using He initialization for better convergence
        if activation == 'relu':
            scale = np.sqrt(2.0 / input_size)  # He initialization
        else:  # tanh
            scale = np.sqrt(1.0 / input_size)  # Xavier initialization
            
        self.W1 = np.random.randn(input_size, hidden_size) * scale
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        # Initialize parameters for tracking training
        self.loss_history = []
        
    def relu(self, Z):
        return np.maximum(0, Z)
    
    def relu_derivative(self, Z):
        return (Z > 0).astype(float)
    
    def tanh(self, Z):
        return np.tanh(Z)
    
    def tanh_derivative(self, Z):
        return 1 - np.power(np.tanh(Z), 2)
    
    def softmax(self, Z):
        # Shift values for numerical stability
        exp_scores = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    def forward(self, X):
        # First layer
        self.Z1 = np.dot(X, self.W1) + self.b1
        
        # Apply activation function
        if self.activation == 'relu':
            self.A1 = self.relu(self.Z1)
        else:  # tanh
            self.A1 = self.tanh(self.Z1)
        
        # Output layer
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.softmax(self.Z2)
        
        return self.A2
    
    def compute_loss(self, y_pred, y_true):
        # Categorical cross-entropy loss with clip for numerical stability
        m = y_true.shape[0]
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.sum(y_true * np.log(y_pred)) / m
        return loss
    
    def backward(self, X, y, learning_rate=0.01):
        m = X.shape[0]
        
        # Backpropagation to the output layer
        dZ2 = self.A2 - y  # Derivative of softmax + cross-entropy
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        # Backpropagation to the hidden layer
        dA1 = np.dot(dZ2, self.W2.T)
        if self.activation == 'relu':
            dZ1 = dA1 * self.relu_derivative(self.Z1)
        else:  # tanh
            dZ1 = dA1 * self.tanh_derivative(self.Z1)
        
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        # Update weights and biases
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def train(self, X, y, epochs=1000, learning_rate=0.01, batch_size=32, verbose=True):
        m = X.shape[0]
        batches = max(m // batch_size, 1)
        
        for epoch in range(epochs):
            # Shuffle the data
            idx = np.random.permutation(m)
            X_shuffled = X[idx]
            y_shuffled = y[idx]
            
            epoch_loss = 0
            for batch in range(batches):
                start = batch * batch_size
                end = min(start + batch_size, m)
                
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                # Forward pass
                y_pred = self.forward(X_batch)
                
                # Compute loss
                loss = self.compute_loss(y_pred, y_batch)
                epoch_loss += loss
                
                # Backward pass
                self.backward(X_batch, y_batch, learning_rate)
            
            # Record average loss for the epoch
            avg_loss = epoch_loss / batches
            self.loss_history.append(avg_loss)
            
            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
    
    def predict(self, X):
        # Forward pass
        y_pred = self.forward(X)
        
        # Return the class with the highest probability
        return np.argmax(y_pred, axis=1)

# Training hyperparameters
hidden_size = 64
epochs = 1000
learning_rate = 0.01
batch_size = 32

# Train model with ReLU activation
print("\nTraining MLP with ReLU activation...")
start_time = time.time()
mlp_relu = MLP(input_size=X_train.shape[1], hidden_size=hidden_size, 
               output_size=num_classes, activation='relu')
mlp_relu.train(X_train, y_train, epochs=epochs, learning_rate=learning_rate, 
               batch_size=batch_size, verbose=True)
relu_training_time = time.time() - start_time
print(f"Training completed in {relu_training_time:.2f} seconds")

# Train model with Tanh activation
print("\nTraining MLP with Tanh activation...")
start_time = time.time()
mlp_tanh = MLP(input_size=X_train.shape[1], hidden_size=hidden_size, 
               output_size=num_classes, activation='tanh')
mlp_tanh.train(X_train, y_train, epochs=epochs, learning_rate=learning_rate, 
               batch_size=batch_size, verbose=True)
tanh_training_time = time.time() - start_time
print(f"Training completed in {tanh_training_time:.2f} seconds")

# Make predictions
y_pred_relu = mlp_relu.predict(X_test)
y_pred_tanh = mlp_tanh.predict(X_test)

# Convert one-hot encoded y_test to class indices
y_test_indices = np.argmax(y_test, axis=1)

# Evaluate accuracy
accuracy_relu = accuracy_score(y_test_indices, y_pred_relu)
accuracy_tanh = accuracy_score(y_test_indices, y_pred_tanh)

print(f"\nReLU Accuracy: {accuracy_relu:.4f}")
print(f"Tanh Accuracy: {accuracy_tanh:.4f}")

# Plot loss convergence
plt.figure(figsize=(10, 6))
plt.plot(mlp_relu.loss_history, label='ReLU')
plt.plot(mlp_tanh.loss_history, label='Tanh')
plt.title('Loss Convergence: ReLU vs Tanh')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('d:/Pythn/loss_convergence.png')
plt.show()

# Use PCA to reduce to 2D for visualization
pca = PCA(n_components=2)
X_test_2d = pca.fit_transform(X_test)

# Plot decision boundaries using PCA-transformed data
def plot_decision_boundary(X_2d, y, model, activation_name):
    h = 0.02  # Step size in the mesh
    
    # Set min and max values with margin
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    
    # Generate a grid of points
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Convert 2D grid to original feature space
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Inverse transform to get back to original dimensions
    grid_orig = pca.inverse_transform(grid)
    
    # Predict classes for each point in the grid
    Z = model.predict(grid_orig)
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.tab10)
    
    # Plot the training points
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_test_indices, 
                          edgecolors='k', alpha=0.6, cmap=plt.cm.tab10)
    plt.title(f'Decision Boundary with {activation_name} Activation')
    plt.xlabel('PCA Feature 1')
    plt.ylabel('PCA Feature 2')
    
    # Add a colorbar and set ticks to class names
    colorbar = plt.colorbar(scatter, ticks=range(len(class_names)))
    colorbar.set_label('Class')
    colorbar.set_ticklabels(class_names)
    
    plt.grid(True)
    plt.savefig(f'd:/Pythn/decision_boundary_{activation_name.lower()}.png')
    plt.show()

# Plot decision boundaries for both models
plot_decision_boundary(X_test_2d, y_test, mlp_relu, 'ReLU')
plot_decision_boundary(X_test_2d, y_test, mlp_tanh, 'Tanh')

# Print classification reports
print("\nClassification Report for ReLU:")
print(classification_report(y_test_indices, y_pred_relu, target_names=class_names))

print("\nClassification Report for Tanh:")
print(classification_report(y_test_indices, y_pred_tanh, target_names=class_names))

# Plot confusion matrices
plt.figure(figsize=(16, 7))

plt.subplot(1, 2, 1)
cm_relu = confusion_matrix(y_test_indices, y_pred_relu)
plt.imshow(cm_relu, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix - ReLU')
plt.colorbar()
tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, class_names, rotation=90)
plt.yticks(tick_marks, class_names)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')

plt.subplot(1, 2, 2)
cm_tanh = confusion_matrix(y_test_indices, y_pred_tanh)
plt.imshow(cm_tanh, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Tanh')
plt.colorbar()
plt.xticks(tick_marks, class_names, rotation=90)
plt.yticks(tick_marks, class_names)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')

plt.savefig('d:/Pythn/confusion_matrices.png')
plt.show()

# Analyze and compare the two activation functions
print("\n----- Comparison of ReLU and Tanh Activation Functions -----")
print(f"1. Loss Convergence Rate:")
final_loss_relu = mlp_relu.loss_history[-1]
final_loss_tanh = mlp_tanh.loss_history[-1]
print(f"   - Final loss for ReLU: {final_loss_relu:.4f}")
print(f"   - Final loss for Tanh: {final_loss_tanh:.4f}")
print(f"   - {'ReLU' if final_loss_relu < final_loss_tanh else 'Tanh'} achieved lower final loss")

print(f"\n2. Training Time:")
print(f"   - ReLU: {relu_training_time:.2f} seconds")
print(f"   - Tanh: {tanh_training_time:.2f} seconds")
print(f"   - {'ReLU' if relu_training_time < tanh_training_time else 'Tanh'} was faster to train")

print(f"\n3. Classification Accuracy:")
print(f"   - ReLU: {accuracy_relu:.4f}")
print(f"   - Tanh: {accuracy_tanh:.4f}")
print(f"   - {'ReLU' if accuracy_relu > accuracy_tanh else 'Tanh'} has higher accuracy")

print("\n4. Decision Boundaries:")
print("   - See the plotted decision boundaries for visual comparison")
print("   - Key differences include separation clarity and boundary smoothness")
