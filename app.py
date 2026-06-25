# app.py

import streamlit as st
import pandas as pd
import pickle

# ==========================
# Load Models
# ==========================

rf_model = pickle.load(open("churn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
cluster_model = pickle.load(open("cluster_model.pkl", "rb"))
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))

# ==========================
# Page Config
# ==========================

st.set_page_config(
    page_title="Customer Segmentation & Churn Prediction",
    layout="wide"
)

st.title("Customer Segmentation & Churn Prediction")

st.write(
    "Predict customer churn and identify customer segments using Machine Learning."
)

# ==========================
# Sidebar Inputs
# ==========================

st.sidebar.header("Customer Information")

year_birth = st.sidebar.number_input(
    "Year Birth",
    min_value=1940,
    max_value=2025,
    value=1985
)

education = st.sidebar.number_input(
    "Education (Encoded)",
    min_value=0,
    max_value=10,
    value=2
)

marital = st.sidebar.number_input(
    "Marital Status (Encoded)",
    min_value=0,
    max_value=10,
    value=1
)

income = st.sidebar.number_input(
    "Income",
    min_value=0,
    value=50000
)

kidhome = st.sidebar.number_input(
    "Kidhome",
    min_value=0,
    max_value=5,
    value=0
)

teenhome = st.sidebar.number_input(
    "Teenhome",
    min_value=0,
    max_value=5,
    value=0
)

recency = st.sidebar.number_input(
    "Recency",
    min_value=0,
    max_value=100,
    value=30
)

mnt_wines = st.sidebar.number_input(
    "Wine Spending",
    min_value=0,
    value=100
)

mnt_fruits = st.sidebar.number_input(
    "Fruit Spending",
    min_value=0,
    value=20
)

mnt_meat = st.sidebar.number_input(
    "Meat Spending",
    min_value=0,
    value=50
)

mnt_fish = st.sidebar.number_input(
    "Fish Spending",
    min_value=0,
    value=20
)

mnt_sweet = st.sidebar.number_input(
    "Sweet Spending",
    min_value=0,
    value=10
)

mnt_gold = st.sidebar.number_input(
    "Gold Spending",
    min_value=0,
    value=20
)

deals = st.sidebar.number_input(
    "Deals Purchases",
    min_value=0,
    value=2
)

web = st.sidebar.number_input(
    "Web Purchases",
    min_value=0,
    value=4
)

catalog = st.sidebar.number_input(
    "Catalog Purchases",
    min_value=0,
    value=2
)

store = st.sidebar.number_input(
    "Store Purchases",
    min_value=0,
    value=5
)

visits = st.sidebar.number_input(
    "Web Visits Per Month",
    min_value=0,
    value=5
)

days_customer = st.sidebar.number_input(
    "Days Customer",
    min_value=0,
    value=1000
)

customer_age = 2025 - year_birth

# ==========================
# Build Input Data
# ==========================

input_dict = {}

for col in feature_columns:
    input_dict[col] = 0

# Fill available values

if "Year_Birth" in input_dict:
    input_dict["Year_Birth"] = year_birth

if "Education" in input_dict:
    input_dict["Education"] = education

if "Marital_Status" in input_dict:
    input_dict["Marital_Status"] = marital

if "Income" in input_dict:
    input_dict["Income"] = income

if "Kidhome" in input_dict:
    input_dict["Kidhome"] = kidhome

if "Teenhome" in input_dict:
    input_dict["Teenhome"] = teenhome

if "Recency" in input_dict:
    input_dict["Recency"] = recency

if "MntWines" in input_dict:
    input_dict["MntWines"] = mnt_wines

if "MntFruits" in input_dict:
    input_dict["MntFruits"] = mnt_fruits

if "MntMeatProducts" in input_dict:
    input_dict["MntMeatProducts"] = mnt_meat

if "MntFishProducts" in input_dict:
    input_dict["MntFishProducts"] = mnt_fish

if "MntSweetProducts" in input_dict:
    input_dict["MntSweetProducts"] = mnt_sweet

if "MntGoldProds" in input_dict:
    input_dict["MntGoldProds"] = mnt_gold

if "NumDealsPurchases" in input_dict:
    input_dict["NumDealsPurchases"] = deals

if "NumWebPurchases" in input_dict:
    input_dict["NumWebPurchases"] = web

if "NumCatalogPurchases" in input_dict:
    input_dict["NumCatalogPurchases"] = catalog

if "NumStorePurchases" in input_dict:
    input_dict["NumStorePurchases"] = store

if "NumWebVisitsMonth" in input_dict:
    input_dict["NumWebVisitsMonth"] = visits

if "Customer_Age" in input_dict:
    input_dict["Customer_Age"] = customer_age

if "Days_Customer" in input_dict:
    input_dict["Days_Customer"] = days_customer

features = pd.DataFrame([input_dict])

# ==========================
# Prediction
# ==========================

if st.button("Predict"):

    scaled = scaler.transform(features)

    churn = rf_model.predict(scaled)[0]
    segment = cluster_model.predict(scaled)[0]

    st.subheader("Prediction Results")

    if churn == 1:
        st.error("⚠ Customer likely to churn")
    else:
        st.success("✅ Customer likely to stay")

    st.info(f"Customer Segment: {segment}")

    st.write("Input Features")

    st.dataframe(features)

# ==========================
# Feature List
# ==========================

with st.expander("Model Features"):
    st.write(feature_columns)