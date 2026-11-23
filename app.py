"""Predict Amazon review sentiment with a trained artificial neural network.

Run with: streamlit run app.py
"""

from pathlib import Path

import joblib
import streamlit as st
from sklearn.feature_extraction.text import HashingVectorizer


MODEL_PATH = Path(__file__).parent / "sentiment_model.joblib"


@st.cache_resource
def load_model(model_path):
    """Load the trained ANN and matching text feature converter."""
    saved_data = joblib.load(model_path)

    vectorizer = HashingVectorizer(
        n_features=saved_data["n_features"],
        alternate_sign=False,
        ngram_range=(1, 2),
        lowercase=True,
        stop_words="english",
        dtype="float32",
    )

    return saved_data["model"], vectorizer


st.set_page_config(
    page_title="Amazon Review ANN Sentiment",
    page_icon="🛍️",
)

st.title("🛍️ Amazon Review Sentiment (ANN)")
st.write(
    "Enter a review to classify its sentiment with an artificial neural network."
)

if not MODEL_PATH.is_file():
    st.error("Trained ANN model not found. Run train.py first.")
    st.stop()

try:
    model, vectorizer = load_model(str(MODEL_PATH))
except Exception as error:
    st.error(f"Could not load the trained ANN model: {error}")
    st.stop()

review = st.text_area(
    "Review text",
    placeholder="Type or paste a product review here...",
    height=160,
)

if st.button("Predict sentiment", type="primary"):
    if not review.strip():
        st.warning("Enter a review first.")
    else:
        review_features = vectorizer.transform([review.strip()])
        prediction = int(model.predict(review_features)[0])

        if prediction == 1:
            st.success("Positive review")
        else:
            st.error("Negative review")

        probabilities = model.predict_proba(review_features)[0]
        st.write(f"Negative: {probabilities[0]:.1%}")
        st.write(f"Positive: {probabilities[1]:.1%}")