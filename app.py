import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Load data
df = pd.read_csv("D:\Car price prediction\CarPrice_Assignment.csv")

# Preprocess data
df["CarName"] = df["CarName"].apply(lambda x: str(x).split(" ")[0].lower())
df["CarName"].replace({"maxda": "mazda", "porcshce": "porsche", "toyouta": "toyota", "vokswagen": "volkswagen", "vw": "volkswagen"}, inplace=True)
df.drop(['symboling', 'car_ID'], axis=1, inplace=True)

# Label encoding for categorical variables
cat = df.select_dtypes(include="object").columns.tolist()
le = LabelEncoder()
for i in cat:
    df[i] = le.fit_transform(df[i])

# Define features and target
x = df.drop("price", axis=1)
y = df["price"]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Train models
model_lr = LinearRegression()
model_lr.fit(x_train, y_train)

model_rf = RandomForestRegressor()
model_rf.fit(x_train, y_train)

# Streamlit UI
st.title("Car Price Prediction App")
st.write("Enter the details below to predict the car price.")

# Create user input fields for prediction
car_data = {}
for column in x.columns:
    if column in cat:
        unique_vals = df[column].unique()
        car_data[column] = st.selectbox(f"{column}", unique_vals)
    else:
        car_data[column] = st.number_input(f"{column}", min_value=float(df[column].min()), max_value=float(df[column].max()), step=0.1)

# Convert input to DataFrame for prediction
input_df = pd.DataFrame([car_data])

# Ensure proper data types for input
for col in cat:
    if col in input_df.columns:
        if input_df[col].dtype == 'object' or input_df[col].apply(lambda x: isinstance(x, str)).any():
            input_df[col] = input_df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else np.nan)

# Check for NaNs introduced by unseen categories
if input_df.isnull().values.any():
    st.error("Error: Some input values are not recognized. Please ensure that all input values are valid.")
else:
    # Make predictions
    if st.button("Predict Price"):
        pred_lr = model_lr.predict(input_df)[0]
        pred_rf = model_rf.predict(input_df)[0]

        # Display results
        st.write(f"*Linear Regression Prediction:* ${pred_lr:.2f}")
        st.write(f"*Random Forest Prediction:* ${pred_rf:.2f}")
