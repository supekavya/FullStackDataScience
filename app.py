import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



# ---------------- Load CSS ----------------
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

st.set_page_config(" Logistic Regression", layout="centered")

# Title
st.markdown("""
<div class="card">
<h1>Logistic Regression</h1>
<p>Predict <b>Heart Disease</b> from <b>Features</b></p>
</div>
""", unsafe_allow_html=True)

# ---------------- Load Data ----------------
@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")

df = load_data()

#Dataset preview
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Prepare Data ----------------
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# Encode categorical columns
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------- Train Model ----------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ---------------- Test Predictions ----------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ---------------- Performance Metrics ----------------
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred)

# ---------------- Performance Section ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Performance")


c1, c2, c3 = st.columns(3)
c1.metric("Accuracy", f"{accuracy:.2f}")
c2.metric("Precision", f"{report['1']['precision']:.2f}")
c3.metric("Recall", f"{report['1']['recall']:.2f}")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Confusion Matrix ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Confusion Matrix")

fig, ax = plt.subplots()
ax.imshow(cm)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center")

st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)
# ---------------- Prediction Section ----------------
st.markdown('<div class="section-title">Make a Prediction</div>', unsafe_allow_html=True)

# Use mean values as base input
user_input = X.mean().copy()

# Example slider (single important feature)
if "Age" in df.columns:
    user_input["Age"] = st.slider(
        "Age",
        int(df["Age"].min()),
        int(df["Age"].max()),
        int(df["Age"].mean())
    )

# Convert to array
user_array = np.array(user_input).reshape(1, -1)
user_scaled = scaler.transform(user_array)

prediction = model.predict(user_scaled)[0]
probability = model.predict_proba(user_scaled)[0][1]

# ---------------- Prediction Output ----------------
result = "Heart Disease" if prediction == 1 else "No Heart Disease"

st.markdown(
    f"""
    <div class="prediction-text">
        {result}<br>
        Probability: {probability:.2f}
    </div>
    """,
    unsafe_allow_html=True
)
