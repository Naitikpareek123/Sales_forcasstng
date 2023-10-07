import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("Pre-Processsing\sales_prediction.csv")  # Update the path to your dataset

# Data Preprocessing
# Fill missing values in 'Item_Weight' with the mean
df['Item_Weight'].fillna(df['Item_Weight'].mean(), inplace=True)

# Encode categorical variables using one-hot encoding
df = pd.get_dummies(df, columns=['Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Item_Identifier'])  # Encoding missing columns

# Extract year and month from 'Outlet_Establishment_Year' and drop the original column
df['Year'] = pd.to_datetime(df['Outlet_Establishment_Year'], format='%Y').dt.year
df['Month'] = pd.to_datetime(df['Outlet_Establishment_Year'], format='%Y').dt.month
df.drop(columns=['Outlet_Establishment_Year'], inplace=True)

# Split the data into features (X) and target (y)
X = df.drop(columns=['Item_Outlet_Sales'])
y = df['Item_Outlet_Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Regressor with Randomized Search CV
param_dist = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestRegressor(random_state=42)
random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=10, cv=5, n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)
best_rf_model = random_search.best_estimator_

# Make predictions on the test set
y_pred = best_rf_model.predict(X_test)

# Calculate MAE and RMSE
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)

# Plot feature importances
feature_importances = best_rf_model.feature_importances_
sorted_indices = np.argsort(feature_importances)[::-1]
sorted_features = X.columns[sorted_indices]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), feature_importances[sorted_indices])
plt.xticks(range(X.shape[1]), sorted_features, rotation=90)
plt.show()

# User input for making predictions
print("\nEnter the following information to predict Item Outlet Sales:")
item_weight = float(input("Item Weight: "))
item_fat_content = int(input("Item Fat Content (0 for Low Fat, 1 for Regular): "))
item_visibility = float(input("Item Visibility: "))
item_type = int(input("Item Type (0-15): "))
item_mrp = float(input("Item MRP: "))
outlet_identifier = int(input("Outlet Identifier (0-9): "))
outlet_size = int(input("Outlet Size (0-2): "))
outlet_location_type = int(input("Outlet Location Type (0-2): "))
outlet_type = int(input("Outlet Type (0-3): "))
year = int(input("Year (e.g., 2019): "))
month = int(input("Month (1-12): "))


# Create a DataFrame with the user's input
# Create a DataFrame with the user's input
user_input = pd.DataFrame({
    'Item_Weight': [item_weight],
    'Item_Fat_Content_Low Fat': [1 if item_fat_content == 0 else 0],  # Update column name
    'Item_Fat_Content_reg': [1 if item_fat_content == 1 else 0],  # Update column name
    'Item_Visibility': [item_visibility],
    'Item_Type_{}'.format(item_type): [1],  # Update column name dynamically
    'Item_MRP': [item_mrp],
    'Outlet_Identifier_{}'.format(outlet_identifier): [1],  # Update column name dynamically
    'Outlet_Size_{}'.format(outlet_size): [1],  # Update column name dynamically
    'Outlet_Location_Type_{}'.format(outlet_location_type): [1],  # Update column name dynamically
    'Outlet_Type_{}'.format(outlet_type): [1],  # Update column name dynamically
    'Year': [year],
    'Month': [month]
})

# Make a prediction using the trained model
predicted_sales = best_rf_model.predict(user_input)

# Print the predicted sales
print("\nPredicted Sales:", predicted_sales[0])

