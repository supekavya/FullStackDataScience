import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# ---- App Configuration ----
st.set_page_config(page_title="Iris Decision Tree", layout="centered")

st.title("🌿 Iris Prediction using Decision Tree")

# ---- Load Data ----
data = load_iris()
features = data.data
labels = data.target

cols = data.feature_names
species = data.target_names

iris_data = pd.DataFrame(features, columns=cols)
iris_data["Label"] = labels
iris_data["Species"] = iris_data["Label"].map(lambda i: species[i])

# ---- Sidebar Settings ----
st.sidebar.title("Tree Parameters")

depth = st.sidebar.selectbox(
    "Select maximum depth",
    options=list(range(1, 11)),
    index=2
)

# ---- Split Dataset ----
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.25, random_state=1
)

# ---- Train Decision Tree ----
tree = DecisionTreeClassifier(max_depth=depth, random_state=1)
tree.fit(X_train, y_train)

# ---- User Inputs ----
st.sidebar.subheader("Enter Flower Details")

user_features = []
for idx, name in enumerate(cols):
    val = st.sidebar.number_input(
        name,
        min_value=float(features[:, idx].min()),
        max_value=float(features[:, idx].max()),
        value=float(features[:, idx].mean())
    )
    user_features.append(val)

sample = np.array([user_features])

# ---- Prediction ----
result = tree.predict(sample)[0]
predicted = species[result]

acc = accuracy_score(y_test, tree.predict(X_test))

# ---- Display ----
st.info(f"🌼 Predicted Species: **{predicted}**")
st.metric("Accuracy", f"{acc * 100:.2f}%")

# ---- Show Data ----
st.subheader("Dataset Overview")
st.dataframe(iris_data, height=250)
