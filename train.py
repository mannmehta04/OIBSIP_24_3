import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import mplcursors

data = pd.read_csv('./data/car_data.csv')
data['Car_Age'] = 2021 - data['Year']

columns_to_drop = ['Car_Name', 'Selling_Price']
for column in columns_to_drop:
    if column not in data.columns:
        print(f"Warning: '{column}' column not found in the dataset.")
columns_to_drop = [column for column in columns_to_drop if column in data.columns]

X = data.drop(columns_to_drop, axis=1)
y = data['Selling_Price'] if 'Selling_Price' in data.columns else None

X = pd.get_dummies(X, drop_first=True)

if y is not None:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    print(f'Training MSE: {train_mse}')
    print(f'Testing MSE: {test_mse}')
    print(f'Training R^2: {train_r2}')
    print(f'Testing R^2: {test_r2}')

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(y_test, y_pred_test, color='blue', label='Predicted vs Actual')
    plt.plot([0, max(y_test)], [0, max(y_test)], color='red', linestyle='--', label='Ideal Fit')
    plt.xlabel('Actual Selling Price')
    plt.ylabel('Predicted Selling Price')
    plt.title('Actual vs Predicted Selling Price')
    plt.legend()

    cursor = mplcursors.cursor(scatter, hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(f'Actual: {y_test.iloc[sel.index]:.2f}\nPredicted: {y_pred_test[sel.index]:.2f}'))

    plt.show()

    residuals = y_test - y_pred_test
    plt.figure(figsize=(10, 6))
    hist = sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')

    cursor = mplcursors.cursor(hist, hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(f'Residual: {residuals.iloc[sel.target.index]:.2f}\nFrequency: {sel.target[1]}'))

    plt.show()
else:
    print("Error: Target variable 'Selling_Price' not found in the dataset.")
