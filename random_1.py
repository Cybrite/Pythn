import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold

data = pd.read_csv('Iris.csv')

data = data.drop(columns=['Id'])

X = data.drop(columns=['Species'])  # Features
y = data['Species']  # Target

model = RandomForestClassifier(random_state=42)

# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
kfold_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

print("K-Fold Cross-Validation:")
print(f"  Accuracies: {kfold_scores}")
print(f"  Mean Accuracy: {kfold_scores.mean():.2f}")

# Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
skfold_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

print("\nStratified K-Fold Cross-Validation:")
print(f"  Accuracies: {skfold_scores}")
print(f"  Mean Accuracy: {skfold_scores.mean():.2f}")
