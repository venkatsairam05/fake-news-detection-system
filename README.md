# Fake News Detection System

## Overview
This project detects whether a news article is Fake or Real using Natural Language Processing (NLP) and Machine Learning.

## Features
- NLP text preprocessing
- TF-IDF vectorization
- Machine Learning classification
- Confidence score prediction
- Explainable AI using SHAP
- Interactive Streamlit web app

## Technologies Used
- Python
- Scikit-learn
- Streamlit
- SHAP
- Pandas
- NumPy

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run model training:

```bash
python train_model.py
```

Run Streamlit app:

```bash
streamlit run app.py
```

## Project Structure

```text
fake-news-detector/
│
├── app.py
├── train_model.py
├── explainability.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
└── data/
```

## Author
Venkat Sai