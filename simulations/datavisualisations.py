# import limitedcatch.csv from ../gfsp_code/data/limitedcatch.csv
# can probably delete this entire file 

import pandas as pd
import matplotlib.pyplot as plt

# Load the data
lim_data = pd.read_csv('gfsp_code/data/LimitedCatch.csv')
spring_data = pd.read_csv('final_code/data/Spring.csv')

# Display the last 5 rows of the data
print(lim_data.tail())
print(spring_data.tail())

# calculate mean catch amount for each dataset
lim_mean = lim_data.mean()
spring_mean = spring_data.mean()

#plot the mean as a solid red line for each dataset
# print mean
print(f"Mean catch amount for Limited Catch dataset: {lim_mean}")
print(f"Mean catch amount for Spring dataset: {spring_mean}")

# Plotting only the second column, scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(range(len(lim_data)), lim_data.iloc[:, 1], marker='o')
plt.axhline(y=lim_mean.iloc[1], color='r', linestyle='-', label=f'Mean: {lim_mean.iloc[1]:.2f}')
plt.title('Limited Catch Data')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid()
plt.show()

# Plot the last column (or 11th column) of the spring data
plt.figure(figsize=(10, 6))
plt.scatter(range(len(spring_data)), spring_data.iloc[:, -1], marker='o')
plt.axhline(y=spring_mean.iloc[-1], color='r', linestyle='-', label=f'Mean: {spring_mean.iloc[-1]:.2f}')
plt.title('Spring Data - Last Column')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid()
plt.show()
