#question-1
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import cross_val_score, KFold
# from sklearn.preprocessing import LabelEncoder

# data = pd.read_csv('Iris.csv')
# data = data.drop(columns=['Id'])

# X = data.drop(columns=['Species'])  # Features
# y = data['Species']  # Target


# le = LabelEncoder()
# y = le.fit_transform(y)

# model = LogisticRegression(max_iter=200, random_state=42)

# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# kfold_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

# print("K-Fold Cross-Validation (Logistic Regression):")
# print(f"  Accuracies: {kfold_scores}")
# print(f"  Mean Accuracy: {kfold_scores.mean():.2f}")


#question 2

# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import cross_val_score, StratifiedKFold
# from sklearn.preprocessing import LabelEncoder

# data = pd.read_csv('Iris.csv')
# data = data.drop(columns=['Id'])

# X = data.drop(columns=['Species'])  # Features
# y = data['Species']  # Target


# le = LabelEncoder()
# y = le.fit_transform(y)

# model = LogisticRegression(max_iter=200, random_state=42)

# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# skfold_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

# print("Stratified K-Fold Cross-Validation (Logistic Regression):")
# print(f"  Accuracies: {skfold_scores}")
# print(f"  Mean Accuracy: {skfold_scores.mean():.2f}")


#question 3

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

file_path = "modlib.data"  
data = pd.read_csv(file_path)

X = data.iloc[:, :-1]  # All columns except the last one
y = data.iloc[:, -1]   # The last column

model = RandomForestClassifier(random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
y_pred = cross_val_predict(model, X, y, cv=kf)

# Calculate evaluation metrics
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, average='weighted')
recall = recall_score(y, y_pred, average='weighted')
f1 = f1_score(y, y_pred, average='weighted')

print("Random Forest Model Evaluation (Libras Dataset):")
print(f"  Accuracy: {accuracy:.2f}")
print(f"  Precision: {precision:.2f}")
print(f"  Recall: {recall:.2f}")
print(f"  F1-Score: {f1:.2f}")
