# model.py

import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# =========================
# Load Dataset
# =========================

df = pd.read_csv("customer_segmentation.csv")

print("Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# =========================
# Handle Missing Values
# =========================

df = df.dropna()

# =========================
# Date Processing
# =========================

if "Dt_Customer" in df.columns:

    df["Dt_Customer"] = pd.to_datetime(
        df["Dt_Customer"],
        dayfirst=True,
        errors="coerce"
    )

    df = df.dropna(subset=["Dt_Customer"])

    # Customer age
    df["Customer_Age"] = 2025 - df["Year_Birth"]

    # Days as customer
    latest_date = df["Dt_Customer"].max()

    df["Days_Customer"] = (
        latest_date - df["Dt_Customer"]
    ).dt.days

# =========================
# Encode Categorical Columns
# =========================

label_encoders = {}

categorical_cols = [
    "Education",
    "Marital_Status"
]

for col in categorical_cols:

    if col in df.columns:

        le = LabelEncoder()

        df[col] = le.fit_transform(
            df[col].astype(str)
        )

        label_encoders[col] = le

# =========================
# Drop Unnecessary Columns
# =========================

drop_cols = []

for col in ["ID", "Dt_Customer"]:
    if col in df.columns:
        drop_cols.append(col)

df = df.drop(columns=drop_cols)

# =========================
# Target Column
# =========================

target = "Response"

if target not in df.columns:
    raise Exception(
        f"Target column '{target}' not found.\n"
        f"Available Columns:\n{df.columns.tolist()}"
    )

# =========================
# Split Features / Target
# =========================

X = df.drop(columns=[target])
y = df[target]

# =========================
# Scaling
# =========================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# =========================
# Random Forest Churn Model
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nRandom Forest Accuracy:", round(accuracy * 100, 2), "%")

# Save model
with open("churn_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

# =========================
# Customer Segmentation
# =========================

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

kmeans.fit(X_scaled)

with open("cluster_model.pkl", "wb") as f:
    pickle.dump(kmeans, f)

# =========================
# Save Feature Names
# =========================

with open("feature_columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("\nFiles Created Successfully:")
print("✓ churn_model.pkl")
print("✓ cluster_model.pkl")
print("✓ scaler.pkl")
print("✓ feature_columns.pkl")