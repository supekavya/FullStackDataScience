import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

# ---- Page Setup ----
st.set_page_config(page_title="Tree Regression", layout="centered")

st.title("📊 Regression using Decision Tree")

# ---- Create Synthetic Data ----
np.random.seed(1)

feature = np.random.uniform(0, 10, size=(120, 1))
target = 3 * feature.flatten() + np.random.normal(0, 3, 120)

dataset = pd.DataFrame({
    "Feature": feature.flatten(),
    "Target": target
})

# ---- Sidebar Options ----
st.sidebar.title("Model Configuration")

depth_value = st.sidebar.number_input(
    "Maximum Tree Depth",
    min_value=1,
    max_value=10,
    value=4
)

# ---- Data Split ----
X_train, X_test, y_train, y_test = train_test_split(
    feature, target, test_size=0.25, random_state=1
)

# ---- Train Model ----
tree_regressor = DecisionTreeRegressor(
    max_depth=depth_value,
    random_state=1
)

tree_regressor.fit(X_train, y_train)

# ---- User Input ----
st.sidebar.subheader("Predict Output")

user_value = st.sidebar.number_input(
    "Enter input value",
    min_value=float(feature.min()),
    max_value=float(feature.max()),
    value=float(feature.mean())
)

sample_input = np.array([[user_value]])

# ---- Prediction & Score ----
estimated = tree_regressor.predict(sample_input)[0]
performance = r2_score(y_test, tree_regressor.predict(X_test))

# ---- Display Results ----
st.info(f"🎯 Predicted Result: **{estimated:.2f}**")
st.metric("R² Score", f"{performance:.2f}")

# ---- Show Dataset ----
st.subheader("Dataset Preview")
st.dataframe(dataset, height=240)
