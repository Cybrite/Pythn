# #question 1
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
# import matplotlib.pyplot as plt

# data = pd.read_csv('diabetes.csv')

# X = data.drop('Outcome', axis=1)
# y = data['Outcome']

# X = X.fillna(X.mean())

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# gnb = GaussianNB()
# gnb.fit(X_train, y_train)


# y_pred = gnb.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# report = classification_report(y_test, y_pred)
# cm = confusion_matrix(y_test,y_pred)

# print(f'Accuracy: {accuracy}')
# print('Classification Report:')
# print(report)
# print("Confusion Matrix:\n",cm)

# plt.figure(figsize=(10, 6))
# data.hist(bins=30, figsize=(20, 15), color='skyblue', edgecolor='black')
# plt.suptitle('Histogram of all features')
# plt.show()


#question 2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv('WineQT.csv')


data['Quality_Label'] = data['quality'].apply(lambda x: 1 if x >= 7 else 0)

X = data.drop(['quality', 'Quality_Label'], axis=1)
y = data['Quality_Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)
print("Confusion Matrix:\n", cm)

data['Quality_Label'].value_counts().plot(kind='bar', figsize=(10, 6))
plt.title('Frequency of Wine Quality')
plt.xlabel('Quality Label')
plt.ylabel('Frequency')
plt.xticks(ticks=[0, 1], labels=['Bad Quality', 'Good Quality'], rotation=0)
plt.show()