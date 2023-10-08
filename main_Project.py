import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
try:
    data = pd.read_csv("Pre-Processsing/sales_prediction.csv")
except FileNotFoundError:
    st.error("File not found. Please check the file path.")
    st.stop()

# Handle missing values
data['Item_Weight'].fillna(data['Item_Weight'].median(), inplace=True)
data['Outlet_Size'].fillna(data['Outlet_Size'].mode()[0], inplace=True)

# Encode categorical variables using LabelEncoder
label_encoders = {}
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

# Streamlit app
st.title("Demand Forecasting App")

# User input section within a form
with st.form("user_inputs"):

    # Create placeholders for user inputs
    item_weight = st.number_input("Item Weight", min_value=0.0)
    item_fat_content = st.selectbox("Item Fat Content (Low Fat/Regular)", ["Low Fat", "Regular"])
    item_visibility = st.number_input("Item Visibility (between 0 and 1)", min_value=0.0, max_value=1.0)
    item_type = st.selectbox("Item Type", data['Item_Type'].unique())
    item_mrp = st.number_input("Item MRP (Maximum Retail Price)", min_value=0.0)
    outlet_identifier = st.selectbox("Outlet Identifier", data['Outlet_Identifier'].unique())
    outlet_establishment_year = st.number_input("Outlet Establishment Year", min_value=1985, max_value=2009)
    outlet_size = st.selectbox("Outlet Size", ["Small", "Medium", "High"])
    outlet_location_type = st.selectbox("Outlet Location Type", data['Outlet_Location_Type'].unique())
    outlet_type = st.selectbox("Outlet Type", data['Outlet_Type'].unique())

    if st.form_submit_button("Submit"):
        # Perform computations with collected user inputs
        with st.spinner("Predicting..."):
            try:
                # Create a DataFrame with user inputs (excluding 'Item_Outlet_Sales')
                user_input_data = pd.DataFrame({
                    'Item_Weight': [item_weight],
                    'Item_Fat_Content': [item_fat_content],
                    'Item_Visibility': [item_visibility],
                    'Item_Type': [item_type],
                    'Item_MRP': [item_mrp],
                    'Outlet_Identifier': [outlet_identifier],
                    'Outlet_Establishment_Year': [outlet_establishment_year],
                    'Outlet_Size': [outlet_size],
                    'Outlet_Location_Type': [outlet_location_type],
                    'Outlet_Type': [outlet_type]
                })

                # Encode categorical variables using one-hot encoding
                user_input_data_encoded = pd.get_dummies(user_input_data, columns=['Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Location_Type', 'Outlet_Type', 'Outlet_Size'])

                # Remove 'Item_Outlet_Sales' from the user input data (if it exists)
                if 'Item_Outlet_Sales' in user_input_data_encoded.columns:
                    user_input_data_encoded = user_input_data_encoded.drop(columns=['Item_Outlet_Sales'])


                # Encode categorical variables using one-hot encoding
                user_input_data_encoded = pd.get_dummies(user_input_data, columns=['Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Location_Type', 'Outlet_Type', 'Outlet_Size'])

                


                # Initialize and train a Random Forest Regressor model
                X = data.drop(columns=['Item_Outlet_Sales'])
                y = data['Item_Outlet_Sales']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model.fit(X_train, y_train)

                # Ensure columns match those used during model training
                model_features = X_train.columns
                user_input_data_encoded = user_input_data_encoded.reindex(columns=model_features, fill_value=0)

                # Use the trained Random Forest Regressor to make predictions for the user input
                sales_predictions = rf_model.predict(user_input_data_encoded)

                # Display the prediction
                st.success(f"Sales Prediction: {sales_predictions[0]:.2f}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
