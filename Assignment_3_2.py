from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the Salary Data CSV file
salary_file_path = r"C:\Ganapathi\UCM_Course_work\Neural_Networks&Deep_learning\Assignment_3\Salary_Data.csv"
df_salary = pd.read_csv(salary_file_path)

# Splitting data into training and test sets (1/3 reserved for testing)
X = df_salary.iloc[:, :-1].values  # Features (Years of Experience)
y = df_salary.iloc[:, -1].values   # Target (Salary)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict using the model
y_pred = model.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)

# Visualization: Scatter plot of training and test data
plt.figure(figsize=(10,5))

# Plot training data
plt.scatter(X_train, y_train, color='blue', label="Training Data")

# Plot test data
plt.scatter(X_test, y_test, color='red', label="Test Data")

# Plot regression line
plt.plot(X_train, model.predict(X_train), color='green', linewidth=2, label="Regression Line")

plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary vs Experience (Train & Test Data)")
plt.legend()
plt.show()

mse