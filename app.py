import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Water Quality Prediction", page_icon="💧")

st.title("💧 Water Quality Potability Prediction")
st.write("This app predicts whether water is safe to drink or not using Random Forest Classification.")

data = pd.read_csv("water_potability.csv")
data.fillna(data.mean(), inplace=True)

X = data.drop("Potability", axis=1)
y = data["Potability"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

st.sidebar.header("Enter Water Quality Values")

ph = st.sidebar.slider("pH", 0.0, 14.0, 7.0)
hardness = st.sidebar.slider("Hardness", 50.0, 350.0, 180.0)
solids = st.sidebar.slider("Solids", 300.0, 60000.0, 20000.0)
chloramines = st.sidebar.slider("Chloramines", 0.0, 15.0, 7.0)
sulfate = st.sidebar.slider("Sulfate", 100.0, 500.0, 300.0)
conductivity = st.sidebar.slider("Conductivity", 100.0, 800.0, 400.0)
organic_carbon = st.sidebar.slider("Organic Carbon", 0.0, 30.0, 15.0)
trihalomethanes = st.sidebar.slider("Trihalomethanes", 0.0, 130.0, 60.0)
turbidity = st.sidebar.slider("Turbidity", 0.0, 10.0, 4.0)

input_data = pd.DataFrame({
    "ph": [ph],
    "Hardness": [hardness],
    "Solids": [solids],
    "Chloramines": [chloramines],
    "Sulfate": [sulfate],
    "Conductivity": [conductivity],
    "Organic_carbon": [organic_carbon],
    "Trihalomethanes": [trihalomethanes],
    "Turbidity": [turbidity]
})

prediction = model.predict(input_data)

st.subheader("Prediction Result")

if prediction[0] == 1:
    st.success("✅ Water is Potable / Safe to Drink")
else:
    st.error("❌ Water is Not Potable / Not Safe")