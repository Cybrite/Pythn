#assingment 2
# # question 1
# import numpy as np

# arr = np.array([1,2,3,4,5,6,7,8,9,10])

# print("Mean is",np.mean(arr))
# print("Standard Deviation is",np.std(arr))


# # question 2 	
# import numpy as np

# matrix = np.array([[1,3,2],[4,5,6],[7,8,9]])

# print("\n Matrix:\n")
# print(matrix)
# print("\n inverse Matrix:\n")
# try:
#     inverse_matrix = np.linalg.inv(matrix)
#     print("Inverse of the matrix:")
#     print(inverse_matrix)
# except np.linalg.LinAlgError:
#     print("The matrix is singular and cannot be inverted.")


# question 3
# import matplotlib.pyplot as plt
# import numpy as np

# x = []
# for i in range(-10, 11):  
#     x.append([i]) 
    
# y = np.square(x)

# plt.plot(x,y)
# plt.show()


# # question 4
# import pandas as pd

# data = {
#     'name': ['Alice', 'Bob', 'Charlie', 'David'],
#     'age': [24, 30, 29, 35],
#     'city': ['New York', 'Chicago','San Francisco', 'New York']
# }
# df = pd.DataFrame(data)


# filtered_df = df[df['age'] > 28]
# print(filtered_df)
