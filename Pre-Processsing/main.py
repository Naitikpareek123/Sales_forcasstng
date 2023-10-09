import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import streamlit as st

# Load the dataset
df = pd.read_csv("C:/Users/91785/Documents/GitHub/Sales_forcasstng/Pre-Processsing/sales_prediction.csv")

# Title
st.title("Demand Forecasting App")

# Sidebar with user inputs
st.sidebar.header("User Inputs")

# Data splitting
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)
random_state = st.sidebar.slider("Random State", 1, 100, 42)

# Data preprocessing
st.subheader("Data Preprocessing")

# Show the first 5 rows of the dataset
if st.checkbox("Show First 5 Rows of Data"):
    st.write(df.head())

# Check for missing values
if st.checkbox("Check Missing Values"):
    st.write(df.isnull().sum())

# Data Visualization
st.subheader("Data Visualization")

# Create scatter plot
st.write("Scatter Plot between Item_MRP and Item_Outlet_Sales")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df, x='Item_MRP', y='Item_Outlet_Sales', ax=ax)
st.pyplot(fig)

# Model Training
st.subheader("Model Training")

# Split the data
X = df.drop(columns=['Item_Outlet_Sales'])
y = df['Item_Outlet_Sales']
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# You can add model training and evaluation code here

# Display model evaluation results

# Show feature importances (if available)

