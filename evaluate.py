import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import mplcursors

new_data = pd.read_csv('./data/car_data.csv')
new_data['Car_Age'] = 2021 - new_data['Year']

columns_to_drop = ['Car_Name', 'Selling_Price']
for column in columns_to_drop:
    if column not in new_data.columns:
        print(f"Warning: '{column}' column not found in the dataset.")
columns_to_drop = [column for column in columns_to_drop if column in new_data.columns]

X_new = new_data.drop(columns_to_drop, axis=1, errors='ignore')
y_new = new_data['Selling_Price'] if 'Selling_Price' in new_data.columns else None

X_new = pd.get_dummies(X_new, drop_first=True)

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

y_pred_new = model.predict(X_new)

if y_new is not None:
    new_mse = mean_squared_error(y_new, y_pred_new)
    new_r2 = r2_score(y_new, y_pred_new)
    print(f'New Data MSE: {new_mse}')
    print(f'New Data R^2: {new_r2}')

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(y_new, y_pred_new, color='blue', label='Predicted vs Actual')
    plt.plot([0, max(y_new)], [0, max(y_new)], color='red', linestyle='--', label='Ideal Fit')
    plt.xlabel('Actual Selling Price')
    plt.ylabel('Predicted Selling Price')
    plt.title('Actual vs Predicted Selling Price on New Data')
    plt.legend()

    cursor = mplcursors.cursor(scatter, hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(f'Actual: {y_new.iloc[sel.index]:.2f}\nPredicted: {y_pred_new[sel.index]:.2f}'))

    plt.show()

    residuals_new = y_new - y_pred_new
    plt.figure(figsize=(10, 6))
    hist = sns.histplot(residuals_new, kde=True)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals on New Data')

    cursor = mplcursors.cursor(hist, hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(f'Residual: {residuals_new.iloc[sel.target.index]:.2f}\nFrequency: {sel.target[1]}'))

    plt.show()
else:
    print('Predicted Selling Prices:')
    print(y_pred_new)
