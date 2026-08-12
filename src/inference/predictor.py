from pathlib import Path
import re
import string

import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "models" / "fake_news_svm.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"


# ============================================================
# Text preprocessing
# Same preprocessing used during model training
# ============================================================

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    """
    Clean news text using the same preprocessing
    pipeline used during model training.
    """

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Tokenize
    words = text.split()

    # Remove stopwords and lemmatize
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# ============================================================
# Load trained artifacts
# ============================================================

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


# ============================================================
# Prediction function
# ============================================================

def predict_news(text: str) -> dict:
    """
    Predict whether a news article is real or fake.
    """

    if not isinstance(text, str):
        raise TypeError("News text must be a string.")

    if not text.strip():
        raise ValueError("News text cannot be empty.")

    # Apply the SAME cleaning used during training
    cleaned_text = clean_text(text)

    if not cleaned_text.strip():
        raise ValueError(
            "The article contains no usable text after preprocessing."
        )

    # Transform using the fitted TF-IDF vectorizer
    text_tfidf = vectorizer.transform([cleaned_text])

    # Predict
    prediction = model.predict(text_tfidf)[0]

    # Decision score
    decision_score = model.decision_function(text_tfidf)[0]

    label = "Fake" if prediction == 1 else "Real"

    return {
        "prediction": label,
        "label": int(prediction),
        "decision_score": float(decision_score)
    }