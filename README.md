# 🔍 AI-Based Fake News Detection Using NLP

A hybrid fake news detection system combining traditional Machine Learning with Large Language Models (LLaMA 3) to detect fake news from any source, topic or region worldwide.

## 📊 Results

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 92.5% |
| Random Forest | 96.7% |
| LinearSVC | 98.0% |
| BERT | 98.65% |
| LLaMA 3 (LLM) | Real-time analysis |

## 📸 Screenshots

![App Interface](screenshots/Screenshot%202026-05-09%20at%204.51.49%20PM.png)

![Result](screenshots/Screenshot%202026-05-09%20at%204.52.22%20PM.png)

## 📁 Dataset
- 51,233 news articles combined from:
  - ISOT Dataset (University of Victoria)
  - GossipCop/PolitiFact Dataset

## 🛠️ Tech Stack
- Python 3.12
- Scikit-learn, NLTK, Transformers (BERT)
- Streamlit (Web App)
- Groq API (LLaMA 3 LLM)

## 🚀 How to Run

1. Install dependencies:
pip install -r requirements.txt

2. Add your Groq API key in .env file:
GROQ_API_KEY=your_key_here

3. Run the app:
streamlit run app.py

## 🎯 Features
- Fast ML model for instant detection
- Advanced LLM analysis with detailed reasoning
- Confidence scores and red flag detection
- Works on any news — Indian, international, recent

## 👨‍💻 Author
Vaibhav Saxena — Vistula University Warsaw (AI Specialization)
