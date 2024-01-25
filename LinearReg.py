import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
X = np.array([[3], [6], [10], [23]])
y = np.array([45, 77, 101, 230])
reg.fit(X, y)

print(reg.coef_)
res = reg.predict([[15]])
print(res)


# Predictions for plotting the regression line
x_pred = np.array([[3], [6], [10], [15], [23]])
y_pred = reg.predict(x_pred)

# Plot the original data points
plt.scatter(X, y, color='blue', label='Original Data Points')

# Plot the regression line
plt.plot(x_pred, y_pred, color='red', linewidth=2, label='Linear Regression Line')

# Highlight the point for x=15
plt.scatter([15], reg.predict([[15]]), color='green', marker='*', s=200, label='Prediction for x=15')

# Labeling the plot
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Visualization')
plt.legend()

# Show the plot
plt.show()
