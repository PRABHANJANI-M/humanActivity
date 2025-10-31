import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# ---------------------------------------------------
# 🧾 Streamlit Page Config
# ---------------------------------------------------
st.set_page_config(page_title="Human Activity Recognition (LSTM)", layout="wide")

st.title("🧠 Human Activity Recognition using LSTM")
st.write("Upload your dataset below to predict human activities using a trained LSTM model.")

# ---------------------------------------------------
# 📦 Load Model and Scaler
# ---------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = load_model("model.h5")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"❌ Error loading model or scaler: {e}")
    st.stop()

# ---------------------------------------------------
# 📂 File Upload
# ---------------------------------------------------
uploaded_file = st.file_uploader("📤 Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(df.head())
        
        # Check if target column exists
        if "Activity" in df.columns:
            X = df.drop("Activity", axis=1)
        else:
            X = df
        
        # ---------------------------------------------------
        # ⚙️ Feature Scaling
        # ---------------------------------------------------
        try:
            X_scaled = scaler.transform(X)
        except AttributeError:
            st.warning("⚠️ Loaded scaler seems to be a NumPy array. Applying manual normalization.")
            if isinstance(scaler, (list, tuple)) and len(scaler) == 2:
                mean_, scale_ = scaler
                X_scaled = (X - mean_) / scale_
            else:
                st.error("❌ Scaler format not recognized. Please re-save 'scaler.pkl' using sklearn StandardScaler.")
                st.stop()

        # ---------------------------------------------------
        # 🔄 Reshape for LSTM
        # ---------------------------------------------------
        # LSTM expects 3D input: (samples, timesteps, features)
        X_scaled = np.array(X_scaled)
        X_scaled = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))

        # ---------------------------------------------------
        # 🔮 Predictions
        # ---------------------------------------------------
        predictions = model.predict(X_scaled)
        predicted_classes = np.argmax(predictions, axis=1)

        # ---------------------------------------------------
        # 🧾 Display Results
        # ---------------------------------------------------
        st.subheader("✅ Prediction Results")
        df_results = df.copy()
        df_results["Predicted_Activity"] = predicted_classes
        st.dataframe(df_results.head(20))

        # Download option
        csv = df_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Predictions as CSV",
            data=csv,
            file_name="predicted_activities.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

else:
    st.info("👆 Upload a CSV file to start prediction.")
