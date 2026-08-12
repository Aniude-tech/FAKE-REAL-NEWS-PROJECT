import requests
import streamlit as st


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered",
)


# ============================================================
# Custom styling
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            text-align: center;
            font-size: 42px;
            font-weight: 700;
            margin-bottom: 5px;
        }

        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #666;
            margin-bottom: 30px;
        }

        .result-box {
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            margin-top: 20px;
        }

        .score {
            font-size: 16px;
            margin-top: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Header
# ============================================================

st.markdown(
    '<div class="main-title">📰 Fake News Detector</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Analyze news text using a machine-learning classification model."
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# API configuration
# ============================================================

API_URL = "http://127.0.0.1:8000/predict"


# ============================================================
# Article input
# ============================================================

st.subheader("Enter News Article")

news_text = st.text_area(
    "Paste the article text below",
    placeholder=(
        "Paste the full news article here..."
    ),
    height=300,
    label_visibility="visible",
)


# ============================================================
# Buttons
# ============================================================

col1, col2 = st.columns(2)

with col1:
    check_article = st.button(
        "🔍 Check Article",
        use_container_width=True,
    )

with col2:
    clear_article = st.button(
        "🗑️ Clear",
        use_container_width=True,
    )


# ============================================================
# Clear input
# ============================================================

if clear_article:
    st.rerun()


# ============================================================
# Prediction
# ============================================================

if check_article:

    # -----------------------------------------
    # Input validation
    # -----------------------------------------

    if not news_text.strip():

        st.warning(
            "Please enter a news article before checking."
        )

        st.stop()

    if len(news_text.strip()) < 30:

        st.warning(
            "The article is too short for a meaningful analysis. "
            "Please provide more text."
        )

        st.stop()

    # -----------------------------------------
    # Call FastAPI
    # -----------------------------------------

    with st.spinner("Analyzing article..."):

        try:

            response = requests.post(
                API_URL,
                json={"text": news_text},
                timeout=30,
            )

            # -----------------------------------------
            # Successful response
            # -----------------------------------------

            if response.status_code == 200:

                result = response.json()

                prediction = result["prediction"]
                score = result["decision_score"]

                st.divider()

                st.subheader("Analysis Result")

                if prediction == "Fake":

                    st.error(
                        "⚠️ FAKE NEWS"
                    )

                    st.write(
                        "The model classified this article as **Fake**."
                    )

                else:

                    st.success(
                        "✅ REAL NEWS"
                    )

                    st.write(
                        "The model classified this article as **Real**."
                    )

                st.metric(
                    "Decision Score",
                    f"{score:.4f}",
                )

                st.caption(
                    "The decision score indicates the model's position "
                    "relative to its classification boundary. "
                    "It is not a probability."
                )

            # -----------------------------------------
            # API error
            # -----------------------------------------

            else:
                if response.status_code == 400:
                    st.warning(
                        "⚠️ The text could not be analyzed. "
                        "Please enter a meaningful news article."
                    )

                else:
                    st.error(
                        f"The prediction API returned an error "
                        f"(status code {response.status_code})."
                    )

        # ---------------------------------------------
        # Connection error
        # ---------------------------------------------

        except requests.exceptions.ConnectionError:

            st.error(
                "❌ Could not connect to the prediction API."
            )

            st.info(
                "Make sure FastAPI is running with:"
            )

            st.code(
                "uvicorn app.backend.main:app --reload"
            )

        # ---------------------------------------------
        # Timeout
        # ---------------------------------------------

        except requests.exceptions.Timeout:

            st.error(
                "⏱️ The request timed out. Please try again."
            )

        # ---------------------------------------------
        # Unexpected error
        # ---------------------------------------------

        except Exception as e:

            st.error(
                f"An unexpected error occurred: {e}"
            )


# ============================================================
# About the model
# ============================================================

st.divider()

with st.expander("ℹ️ About this detector"):

    st.write(
        """
        This application uses a machine-learning pipeline to classify
        news articles as **Real** or **Fake**.

        **Model:** Linear Support Vector Machine (SVM)

        **Text representation:** TF-IDF

        **Preprocessing:** Text cleaning, stopword removal and
        lemmatization.

        **Hyperparameter tuning:** The SVM was tuned using
        cross-validation, with the selected value of `C = 100`.

        The detector is a machine-learning model and its predictions
        should be treated as an automated classification rather than
        definitive proof that a news article is true or false.
        """
    )