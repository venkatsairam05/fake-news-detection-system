import streamlit as st
import joblib

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Title
st.title("📰 Fake News Detection System")

st.markdown(
    """
    This AI system detects whether a news article is **Fake** or **Real**
    using NLP and Machine Learning.
    """
)

# Text input
news = st.text_area(
    "Enter News Article",
    height=250,
    placeholder="Paste news article here..."
)

# Predict button
if st.button("Predict News"):

    if news.strip() == "":
        st.warning("⚠ Please enter news text.")

    else:

        # Transform text
        transformed_news = vectorizer.transform([news])

        # Prediction
        prediction = model.predict(transformed_news)
        #confidence score
        probability = model.predict_proba(transformed_news)

        confidence = max(probability[0]) * 100

        # Output
        st.subheader("Prediction Result")

        if prediction[0] == 0:
            st.error("❌ Fake News")
        else:
            st.success("✅ Real News")

        st.info(f"Confidence Score: {confidence:.2f}%")

# Footer
st.markdown("---")
st.caption("Built using Python, NLP, Scikit-learn, and Streamlit")