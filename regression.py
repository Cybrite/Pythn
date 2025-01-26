import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics  import r2_score,mean_squared_error
df = pd.read_csv('city_population_vs_profit.csv')

new_df = df[['Population of city ($1000)', 'Profit of restaurants ($10K)']]
x = np.array(new_df[['Population of city ($1000)']])
y = np.array(new_df[['Profit of restaurants ($10K)']])

print(x.shape)
print(y.shape)

# plt.scatter(x,y,color='red')
# plt.xlabel('Population of city ($1000)')
# plt.ylabel('Profit of restaurants ($10K)')
# plt.show()

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3333,random_state=0)
regressor = LinearRegression()
# print(x_train)
# print(x_test)

regressor.fit(x_train,y_train)
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue',linewidth=3)
plt.xlabel('Population of city ($1000)')
plt.ylabel('Profit of restaurants ($10K)')
plt.show()
y_pred = regressor.predict(x_test)
print('r2 score',r2_score(y_test,y_pred))
print('mean squared error',mean_squared_error(y_test,y_pred))