import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Loading data
california = fetch_california_housing(as_frame=True)
data = california.frame

# 2. Data cleaning
data.dropna(inplace=True)  
data.drop_duplicates(inplace=True)  

# 3. EDA
print("Basic statistics:")
print(data.describe())
print("Distribution data:\n")
sns.pairplot(data)
plt.show()

# 4. Anomaly Detection
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).sum()
print("\nOutliers:")
print(outliers)

# 5. Predictive Modeling
# Select target column (MedHouseVal) and other columns
target_column = 'MedHouseVal'
X = data.drop(target_column, axis=1)
y = data[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train various models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "Random Forest": RandomForestRegressor(random_state=42)
}

# Evaluate the performance of different models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\n{name} - Model performance:\nMean Squared Error: {mse}\nR^2 Score: {r2}")

# 6. Hyperparameter optimization for the best model (Random Forest in this case)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=models["Random Forest"], param_grid=param_grid,
                           scoring='r2', cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Evaluate the performance of the optimized model
y_pred_best = best_model.predict(X_test)
mse_best = mean_squared_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)

print(f"\nRandom Forest (Optimized) - Optimized model performance:\nMean Squared Error: {mse_best}\nR^2 Score: {r2_best}")

# 7. Detailed summary
summary = {
    "Total number of rows": data.shape[0],
    "Total number of columns": data.shape[1],
    "Number of outliers detected": outliers.sum(),
    "Model performance - Linear Regression": {
        "MSE": mean_squared_error(y_test, models["Linear Regression"].predict(X_test)),
        "R^2": r2_score(y_test, models["Linear Regression"].predict(X_test))
    },
    "Model performance - Ridge": {
        "MSE": mean_squared_error(y_test, models["Ridge"].predict(X_test)),
        "R^2": r2_score(y_test, models["Ridge"].predict(X_test))
    },
    "Model performance - Lasso": {
        "MSE": mean_squared_error(y_test, models["Lasso"].predict(X_test)),
        "R^2": r2_score(y_test, models["Lasso"].predict(X_test))
    },
    "Model performance - Random Forest": {
        "MSE": mean_squared_error(y_test, models["Random Forest"].predict(X_test)),
        "R^2": r2_score(y_test, models["Random Forest"].predict(X_test))
    },
    "Optimized model performance - Random Forest (Optimized)": {
        "MSE": mse_best,
        "R^2": r2_best
    }
}

print("\nDetailed Summary:")
for key, value in summary.items():
    print(f"{key}: {value}")
