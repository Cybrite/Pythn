# import numpy as np 
# from collections import Counter 
# from sklearn.datasets import load_iris 
# from sklearn.model_selection import train_test_split 
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
# # Function to calculate distances between a test point and all training points 
# def calculate_distances(X_train, test_point): 
#     distances = [] 
#     for x_train in X_train: 
#         distance = np.sqrt(np.sum((test_point - x_train) ** 2))  # Euclidean distance 
#         distances.append(distance) 
#     return distances 
# # Function to predict the label of a single test point 
# def predict_single(X_train, y_train, test_point, k): 
#     distances = calculate_distances(X_train, test_point)  # Get distances 
#     sorted_indices = np.argsort(distances)  # Get indices of neighbors sorted by distance 
 
#     k_nearest_labels = [] 
#     for i in range(k):  # Collect the labels of the k nearest neighbors 
#         k_nearest_labels.append(y_train[sorted_indices[i]]) 
 
#     # Find the most common label in the k nearest neighbors 
#     label_counts = Counter(k_nearest_labels) 
#     most_common_label = label_counts.most_common(1)[0][0] 
 
#     return most_common_label 
 
# # Function to predict labels for all test points 
# def predict(X_train, y_train, X_test, k): 
#     predictions = [] 
#     for test_point in X_test:  # Loop through all test points 
#         label = predict_single(X_train, y_train, test_point, k)  # Predict for one test point 
#         predictions.append(label)  # Store the prediction 
#     return predictions 
 
# # Load the Iris dataset 
# iris = load_iris() 
# X = iris.data  # Features 
# y = iris.target  # Labels 
 
# # Split the dataset into training and testing sets 
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
 
# # Set the value of k 
# k = 3 
 
# # Predict the labels for the test set 
# y_pred = predict(X_train, y_train, X_test, k) 
 
# # Calculate accuracy 
# accuracy = accuracy_score(y_test, y_pred) 
# print(f"Accuracy: {accuracy * 100:.2f}%\n") 
 
# # Generate and display the classification report 
# print("Classification Report:") 
# print(classification_report(y_test, y_pred, target_names=iris.target_names)) 
 
# # Generate and display the confusion matrix 
# print("Confusion Matrix:") 
# print(confusion_matrix(y_test, y_pred)) 


