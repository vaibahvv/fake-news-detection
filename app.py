import streamlit as st
import pickle
import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Load saved model and vectorizer
with open('models/svc_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered"
)

# Header
st.title("🔍 Fake News Detector")
st.subheader("AI-Based Fake News Detection Using NLP")
st.markdown("---")

# Stats row
col1, col2, col3 = st.columns(3)
col1.metric("Model Accuracy", "98%")
col2.metric("Articles Trained On", "51,233")
col3.metric("Models Compared", "4")

st.markdown("---")

# Detection mode
mode = st.radio("Select Detection Mode:", 
    ["ML Model (Fast)", "LLM Analysis (Advanced)"],
    horizontal=True)

st.subheader("Enter News Article")
title = st.text_input("News Title", placeholder="Enter the headline here...")
text = st.text_area("News Text", height=200, placeholder="Paste the full article text here...")

if st.button("🔍 Detect", use_container_width=True):
    if title and text:
        st.markdown("---")
        st.subheader("Result")

        if mode == "ML Model (Fast)":
            with st.spinner("Analyzing with ML model..."):
                combined = title + ' ' + text
                transformed = tfidf.transform([combined])
                prediction = model.predict(transformed)[0]
                decision = model.decision_function(transformed)[0]
                confidence = round(min(abs(decision) * 20, 99), 1)

            if prediction == "FAKE":
                st.error("FAKE NEWS DETECTED!")
                st.progress(int(confidence))
                st.write(f"Confidence: {confidence}%")
                st.warning("This article shows patterns commonly found in fake news.")
            else:
                st.success("REAL NEWS!")
                st.progress(int(confidence))
                st.write(f"Confidence: {confidence}%")
                st.info("This article shows patterns consistent with real news.")

        else:
            with st.spinner("Analyzing with LLM... (this may take a few seconds)"):
                prompt = f"""You are a fake news detection expert. Analyze this news article and determine if it is FAKE or REAL news.

Title: {title}
Text: {text}

Provide your analysis in this exact format:
VERDICT: [FAKE or REAL]
CONFIDENCE: [percentage like 85%]
REASON: [2-3 sentences explaining why]
RED FLAGS: [list any suspicious elements or write "None"]"""

                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}]
                )
                
                result = response.choices[0].message.content

            if "VERDICT: FAKE" in result:
                st.error("FAKE NEWS DETECTED!")
            else:
                st.success("REAL NEWS!")
            
            st.markdown("### Detailed Analysis")
            st.write(result)

    else:
        st.warning("Please enter both a title and text!")

# Footer
st.markdown("---")
st.caption("Built with Python, Scikit-learn, BERT & LLaMA 3 | Thesis Project")