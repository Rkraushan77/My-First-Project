import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier

# Page configuration (must be first Streamlit command)
st.set_page_config(page_title="Weather Classifier", layout="wide")

# Title and description
st.title("🌦️ K-Nearest Neighbor Weather Classification")
st.markdown("> Hello Everyone, so let's proceed.")
st.markdown("""
_This app uses **K-Nearest Neighbors (KNN)** to classify weather conditions
based on **temperature** and **humidity**._
""")

# Training data
X = np.array([
    [30, 70],
    [25, 80],
    [27, 60],
    [31, 65],
    [23, 85],
    [28, 75]
])

y = np.array([0, 1, 0, 0, 1, 1])

# Label mapping
label_map = {
    0: "Sunny",
    1: "Rainy"
}

# Sidebar input
st.sidebar.header("📊 Input Parameters")

temperature = st.sidebar.slider(
    "Temperature (°C)", 20, 35, 26, 1
)

humidity = st.sidebar.slider(
    "Humidity (%)", 50, 90, 78, 1
)

k = st.sidebar.slider(
    "K (Number of Neighbors)", 1, len(X), 3, 1
)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X, y)

# Prediction
new_data = np.array([[temperature, humidity]])
prediction = knn.predict(new_data)[0]
probabilities = knn.predict_proba(new_data)[0]

weather = label_map[prediction]
confidence = probabilities[prediction] * 100

# Sidebar result
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Prediction Result")

if prediction == 0:
    st.sidebar.success(f"**Weather: {weather}** ☀️")
else:
    st.sidebar.info(f"**Weather: {weather}** 🌧️")

st.sidebar.metric("Confidence", f"{confidence:.1f}%")

# Layout
col1, col2 = st.columns(2)

# Visualization
with col1:
    st.subheader("📈 Classification Visualization")

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        X[y == 0, 0], X[y == 0, 1],
        label="Sunny", s=100, alpha=0.7
    )

    ax.scatter(
        X[y == 1, 0], X[y == 1, 1],
        label="Rainy", s=100, alpha=0.7
    )

    ax.scatter(
        temperature, humidity,
        marker="*", s=300,
        label=f"New Day: {weather}",
        zorder=5
    )

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Humidity (%)")
    ax.set_title("Weather Classification using KNN")
    ax.legend()
    ax.grid(True)
    ax.set_xlim(20, 35)
    ax.set_ylim(50, 90)

    st.pyplot(fig)

# Model info
with col2:
    st.subheader("📋 Model Information")

    st.write("**Training Data Summary**")
    st.write(f"- Total samples: {len(X)}")
    st.write(f"- Sunny days: {np.sum(y == 0)}")
    st.write(f"- Rainy days: {np.sum(y == 1)}")
    st.write(f"- K value: {k}")

    st.markdown("---")

    st.write("**Current Input**")
    st.write(f"- Temperature: **{temperature}°C**")
    st.write(f"- Humidity: **{humidity}%**")

    st.markdown("---")

    st.write("**Prediction Probabilities**")
    st.metric("Sunny Probability", f"{probabilities[0]*100:.1f}%")
    st.metric("Rainy Probability", f"{probabilities[1]*100:.1f}%")

# Footer
st.markdown("---")
st.caption("KNN Weather Classification Model")

# KNN explanation
with st.expander("ℹ️ What is KNN and how it works?"):
    st.markdown("""
**❓ What is KNN?**  
K-Nearest Neighbor is a **supervised machine learning algorithm** used for
classification and regression.

**⚙️ How it works:**
1. Stores all training data
2. Calculates distance to all points
3. Selects the **K nearest neighbors**
4. Uses majority voting for classification

**🌤️ In this model:**
- High temperature + Low humidity → **Sunny**
- Low temperature + High humidity → **Rainy**
""")
