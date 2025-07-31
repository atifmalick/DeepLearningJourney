import streamlit as st
import matplotlib.pyplot as plt

try:
    from transformers import pipeline
except ImportError:
    st.error("❌ Transformers library not properly installed or incompatible.")
    st.stop()

# Page config
st.set_page_config(page_title="Hate Speech Detector", layout="centered")
st.title("🧠 Zero-shot Hate Speech Detection")
st.markdown("Detect whether a text contains **hate speech**, is **offensive**, or is **neutral** using zero-shot learning.")

# Model selection
model_choice = st.radio("Choose Model:", ["English (BART)", "Multilingual (XLM-Roberta)"])

# Load model with caching
@st.cache_resource
def load_model(name):
    return pipeline("zero-shot-classification", model=name)

if model_choice == "English (BART)":
    model_name = "facebook/bart-large-mnli"
else:
    model_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"  # Multilingual zero-shot compatible

model = load_model(model_name)

# Input field
text = st.text_area("✍️ Enter your text here:", placeholder="Type your sentence or speech...", height=150)

# Labels
labels = ["hate speech", "offensive", "neutral"]

# Detect button
if st.button("🚀 Detect"):
    if not text.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("🔍 Analyzing..."):
            result = model(text, labels)

        # Show results
        st.subheader("📊 Prediction Scores")
        for label, score in zip(result["labels"], result["scores"]):
            st.markdown(f"**{label.title()}**: `{score:.4f}`")

        # Plot
        fig, ax = plt.subplots()
        ax.bar(result["labels"], result["scores"], color=["red", "orange", "green"])
        ax.set_ylabel("Confidence Score")
        ax.set_ylim(0, 1)
        ax.set_title("Classification Results")
        st.pyplot(fig)
