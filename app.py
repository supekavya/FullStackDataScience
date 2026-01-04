import streamlit as st
import seaborn as  sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Page config
st.set_page_config("Ridge and Lasso Regression", layout="centered")

def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css("style.css")

# Title
st.markdown("""
<div class="card">
<h1>Ridge & Lasso Regression</h1>
<p>Predict <b>Tip Amount</b> from <b>Total Bill</b></p>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return sns.load_dataset("tips")
data = load_data()

#Dataset preview
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(data.head())
st.markdown('</div>', unsafe_allow_html=True)

X = data[["total_bill"]]
y = data["tip"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Train models
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)

model_choice = st.selectbox(
    "Choose Regression Model",
    ["Ridge Regression", "Lasso Regression"]
)

if model_choice == "Ridge Regression":
    model=ridge_model
    color = "green"
else:
    model=lasso_model
    color = "orange"
    
# Predictions
y_pred = model.predict(X_test)

# Evaluation Metrics
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
adjusted_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Total Bill vs Tip Amount")

fig,ax = plt.subplots()
ax.scatter(data["total_bill"], data["tip"], alpha=0.6)

x_range = data[["total_bill"]]
x_scaled = scaler.transform(x_range)
ax.plot(data["total_bill"], model.predict(x_scaled), color=color, linewidth=2, label=model_choice)

ax.set_xlabel("Total Bill")
ax.set_ylabel("Tip Amount")
ax.legend()

st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Evaluation Metrics")

c1,c2 = st.columns(2)
c1.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
c2.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")

c3,c4 = st.columns(2)
c3.metric("R² Score", f"{r2:.2f}")
c4.metric("Adjusted R² Score", f"{adjusted_r2:.2f}")

st.markdown('</div>', unsafe_allow_html=True)

#prediction
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Make a Prediction")

bill = st.slider(
    "Total Bill ($)",
    float(data["total_bill"].min()),
    float(data["total_bill"].max()),
    float(data["total_bill"].mean())
)

tip = model.predict(scaler.transform([[bill]]))[0]

st.markdown(
    f'<div class="prediction-box">Predicted Tip: ${tip:.2f}</div>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)

