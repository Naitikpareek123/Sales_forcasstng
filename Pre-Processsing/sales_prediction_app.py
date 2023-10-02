import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import streamlit as st

# Load the dataset
df = pd.read_csv(r"C:\Users\91785\Documents\GitHub\Sales_forcasstng\Pre-Processsing\sales_prediction.csv")

# Title
st.title("Sales Prediction Data Visualization")

# Sidebar with user inputs
st.sidebar.header("User Inputs")

# Data preprocessing
st.sidebar.subheader("Data Preprocessing")

# Show the first 5 rows of the dataset
if st.sidebar.checkbox("Show First 5 Rows of Data"):
    st.write(df.head())

# Check for missing values
if st.sidebar.checkbox("Check Missing Values"):
    st.write(df.isnull().sum())

# Data Visualization
st.sidebar.subheader("Data Visualization")

# Create scatter plot
st.write("Scatter Plot between Item_MRP and Item_Outlet_Sales")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df, x='Item_MRP', y='Item_Outlet_Sales', ax=ax)
st.pyplot(fig)

# Histogram and Boxplot of Item_Weight
st.write("Histogram and Boxplot of Item_Weight")
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(data=df, x='Item_Weight', ax=ax[0])
sns.boxplot(data=df, y='Item_Weight', ax=ax[1])
st.pyplot(fig)

# Other visualizations and features can be added similarly

# Data Splitting
st.sidebar.subheader("Data Splitting")

# User inputs for data splitting
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)
random_state = st.sidebar.slider("Random State", 1, 100, 42)

# Split the data
X = df.drop(columns=['Item_Outlet_Sales'])
y = df['Item_Outlet_Sales']
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Show the shape of the training and testing sets
st.write(f"Shape of X_train: {X_train.shape}")
st.write(f"Shape of X_test: {X_test.shape}")
st.write(f"Shape of Y_train: {Y_train.shape}")
st.write(f"Shape of Y_test: {Y_test.shape}")
