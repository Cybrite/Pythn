from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing  import StandardScaler
import pandas as pd

data = pd.read_csv("heart.csv")

data.fillna(data.median(),inplace=True)

scaler = StandardScaler()
features = data.drop(columns=["target"])
labels = data["target"]
features_scaled = scaler.fit_transform(features)

X_train,X_test,y_train,y_test = train_test_split(features_scaled,labels,test_size=0.3,random_state=42)

model = GaussianNB()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print("Classification Report:\n",classification_report(y_test,y_pred))
cm = confusion_matrix(y_test,y_pred)
print("Confusion Matrix:\n",cm)