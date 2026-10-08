import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv('Naive-Bayes-Classification-Data.csv')

# Display basic information about the dataset
print("Dataset Information:")
print(data.head())
print("\nDataset Shape:", data.shape)
print("\nClass Distribution:")
print(data['diabetes'].value_counts())

# Separate features and target variable
X = data[['glucose', 'bloodpressure']]
y = data['diabetes']

# Split the dataset into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# Create and train the Gaussian Naive Bayes classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = nb_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\n{'='*50}")
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print(f"{'='*50}")

# Display confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))

# Example: Predict for new data
print("\n" + "="*50)
print("Example Predictions:")
print("="*50)
new_data = np.array([[45, 85], [60, 65], [30, 75]])
predictions = nb_classifier.predict(new_data)

for i, (glucose, bp) in enumerate(new_data):
    result = "Diabetes" if predictions[i] == 1 else "No Diabetes"
    print(f"Glucose: {glucose}, Blood Pressure: {bp} → Prediction: {result}")

# Calculate prior probabilities
print("\n" + "="*50)
print("Prior Probabilities:")
print("="*50)
print(f"P(No Diabetes) = {nb_classifier.class_prior_[0]:.4f}")
print(f"P(Diabetes) = {nb_classifier.class_prior_[1]:.4f}")
