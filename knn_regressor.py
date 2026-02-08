import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

# ---- Page Setup ----
st.set_page_config(page_title="Regression with KNN", layout="centered")

st.title("🔢 KNN Based Regression Demo")

# ---- Create Sample Data ----
np.random.seed(1)

feature = np.random.uniform(0, 10, size=(120, 1))
target = 2.5 * feature.flatten() + np.random.normal(0, 2, 120)

data = pd.DataFrame({
    "Feature": feature.flatten(),
    "Target": target
})

# ---- Sidebar Configuration ----
st.sidebar.title("Parameters")

neighbors = st.sidebar.number_input(
    "Choose number of neighbors",
    min_value=1,
    max_value=15,
    value=4
)

# ---- Split Data ----
X_train, X_test, y_train, y_test = train_test_split(
    feature, target, test_size=0.25, random_state=1
)

# ---- Train Regressor ----
knn_reg = KNeighborsRegressor(n_neighbors=neighbors)
knn_reg.fit(X_train, y_train)

# ---- User Input ----
st.sidebar.subheader("Make Prediction")

user_x = st.sidebar.number_input(
    "Input feature value",
    min_value=float(feature.min()),
    max_value=float(feature.max()),
    value=float(feature.mean())
)

sample_input = np.array([[user_x]])

# ---- Predict & Evaluate ----
pred_value = knn_reg.predict(sample_input)[0]
score = r2_score(y_test, knn_reg.predict(X_test))

# ---- Display Output ----
st.info(f"📌 Estimated Output: **{pred_value:.2f}**")
st.metric("R² Performance", f"{score:.2f}")

# ---- Show Data ----
st.subheader("Generated Dataset")
st.dataframe(data, height=240)
