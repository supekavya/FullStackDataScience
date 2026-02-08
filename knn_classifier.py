import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# App configuration
st.set_page_config(page_title="Iris KNN Predictor", layout="centered")

st.title("🌼 Iris Flower Prediction using KNN")

# ---- Load and prepare data ----
data = load_iris()
features = data.data
labels = data.target

columns = data.feature_names
flower_types = data.target_names

iris_df = pd.DataFrame(features, columns=columns)
iris_df["Label"] = labels
iris_df["Species"] = iris_df["Label"].map(lambda i: flower_types[i])

# ---- Sidebar options ----
st.sidebar.title("Model Configuration")

neighbors = st.sidebar.select_slider(
    "Select K value",
    options=list(range(1, 16)),
    value=3
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.25, random_state=1
)

# Build model
knn = KNeighborsClassifier(n_neighbors=neighbors)
knn.fit(X_train, y_train)

# ---- User input section ----
st.sidebar.subheader("Enter Flower Measurements")

user_values = []
for idx, name in enumerate(columns):
    val = st.sidebar.number_input(
        label=name,
        min_value=float(features[:, idx].min()),
        max_value=float(features[:, idx].max()),
        value=float(features[:, idx].mean())
    )
    user_values.append(val)

sample = np.array([user_values])

# ---- Prediction ----
result = knn.predict(sample)[0]
predicted_flower = flower_types[result]

test_predictions = knn.predict(X_test)
score = accuracy_score(y_test, test_predictions)

# ---- Display results ----
st.info(f"🌿 Predicted Species: **{predicted_flower}**")
st.metric("Model Accuracy", f"{score * 100:.2f}%")

# ---- Dataset preview ----
st.subheader("Dataset Preview")
st.dataframe(iris_df, height=250)
