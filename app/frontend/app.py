import requests
import streamlit as st


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)


# ============================================================
# Title
# ============================================================

st.title("📰 Fake News Detector")

st.write(
    "Enter a news article below and our machine-learning model "
    "will classify it as Real or Fake."
)


# ============================================================
# API configuration
# ============================================================

API_URL = "http://127.0.0.1:8000/predict"


# ============================================================
# Input
# ============================================================

news_text = st.text_area(
    "News Article",
    placeholder="Paste the news article here...",
    height=300
)


# ============================================================
# Prediction
# ============================================================

if st.button("🔍 Check Article", use_container_width=True):

    if not news_text.strip():
        st.warning("Please enter a news article first.")

    else:

        with st.spinner("Analyzing article..."):

            try:

                response = requests.post(
                    API_URL,
                    json={"text": news_text},
                    timeout=30
                )

                if response.status_code == 200:

                    result = response.json()

                    prediction = result["prediction"]
                    score = result["decision_score"]

                    st.divider()

                    if prediction == "Fake":

                        st.error("⚠️ FAKE NEWS")

                    else:

                        st.success("✅ REAL NEWS")

                    st.write(
                        f"**Decision score:** `{score:.4f}`"
                    )

                else:

                    st.error(
                        f"API error: {response.status_code}"
                    )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to the Fake News Detector API. "
                    "Make sure FastAPI is running."
                )

            except requests.exceptions.Timeout:

                st.error(
                    "The request timed out. Please try again."
                )

            except Exception as e:

                st.error(
                    f"An unexpected error occurred: {e}"
                )