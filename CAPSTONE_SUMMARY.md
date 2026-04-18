# Fake News Detection Assistant

## Domain
News credibility and misinformation detection

## User
Students, readers, journalists, and researchers who want a quick first-pass credibility check for article content.

## Problem
Digital platforms make it easy for false or misleading stories to spread before human fact-checkers can respond. A lightweight machine learning assistant can help users triage suspicious articles and prioritize which ones need deeper verification.

## Success Criteria
- Train a reliable text classification model on the provided dataset
- Expose predictions through a simple web interface
- Show confidence output for transparency
- Keep the project easy to retrain, test, and deploy

## Features
- Uses both headline and article body as model input
- Trains a TF-IDF + Logistic Regression pipeline
- Saves reusable model artifacts
- Displays prediction confidence in Streamlit
- Includes a metrics report for documentation

## Tech Stack
- Python
- Pandas
- scikit-learn
- Joblib
- Streamlit

## Unique Points
- Single shared prediction pipeline for training and deployment
- Reproducible project structure instead of one-off notebook logic
- Streamlit-ready packaging for capstone demonstration

## Future Improvements
- Add explainability with top contributing words
- Include source-domain features and URL reputation checks
- Add multilingual and transformer-based models

