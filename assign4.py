# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# np.random.seed(42)
# num_samples = 400

# data = {
#     "User ID": np.arange(1, num_samples + 1),
#     "Gender": np.random.choice(["Male", "Female"], size=num_samples),
#     "Age": np.random.randint(18, 60, size=num_samples),
#     "Estimated Salary": np.random.randint(15000, 100000, size=num_samples),
#     "Purchased": np.random.choice([0, 1], size=num_samples, p=[0.6, 0.4]),
# }
# df = pd.DataFrame(data)

# label_encoder = LabelEncoder()
# df["Gender"] = label_encoder.fit_transform(df["Gender"])

# x = df[["Gender", "Age", "Estimated Salary"]]
# y = df["Purchased"]

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# model = LogisticRegression()
# model.fit(x_train, y_train)

# y_pred = model.predict(x_test)

# accuracy = accuracy_score(y_test, y_pred)
# conf_matrix = confusion_matrix(y_test, y_pred)
# classification_rep = classification_report(y_test, y_pred)

# print("Accuracy:", accuracy)
# print("Confusion Matrix:\n", conf_matrix)
# print("Classification Report:\n", classification_rep)


# sample_input = np.array([[1, 30, 50000]])
# sample_input = pd.DataFrame(sample_input, columns=["Gender", "Age", "Estimated Salary"])
# sample_input_scaled = scaler.transform(sample_input)
# sample_prediction = model.predict(sample_input_scaled)

# print("Sample Prediction (1=Will Purchase, 0=Will Not Purchase):", sample_prediction[0])



#question 2



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
df = pd.read_csv('Fish.csv')
new_df = df[['Species', 'Weight']]

unique_species = new_df['Species'].unique()  
species_to_num = {species: idx for idx, species in enumerate(unique_species)}
new_df['SpeciesNumeric'] = new_df['Species'].map(species_to_num)


x = np.array(new_df['SpeciesNumeric']).reshape(-1, 1)
y = np.array(new_df['Weight'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3333, random_state=0)


regressor = LinearRegression()
regressor.fit(x_train, y_train)


y_pred = regressor.predict(x_test)


plt.scatter(x_test, y_test, color='green')
plt.plot(np.sort(x_test, axis=0), regressor.predict(np.sort(x_test, axis=0)), color='red', linewidth=3)


plt.xticks(ticks=np.arange(len(unique_species)), labels=unique_species)


plt.xlabel('Species')
plt.ylabel('Weight')
plt.show()


print('r2 score:', r2_score(y_test, y_pred))
print('Mean squared error:', mean_squared_error(y_test, y_pred))
