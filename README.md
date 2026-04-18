# Fake News Detection Assistant

## Problem Statement
Misinformation spreads quickly online, and readers often do not have time to manually verify every article they encounter. This capstone project builds a machine learning assistant that classifies article content as `REAL` or `FAKE` using the supplied fake-news dataset.

## What This Project Includes
- A reusable training and inference package in `fake_news_detection/`
- A reproducible training script: `train_model.py`
- A prediction smoke test: `test_model.py`
- A Streamlit web app: `streamlit_app.py`
- Deployment-ready configuration for Streamlit
- Saved model artifacts and metrics in `artifacts/`

## Tech Stack
- Python
- Pandas
- scikit-learn
- Joblib
- Streamlit

## Project Structure
```text
Fake-News-Detection/
|-- Data/
|-- artifacts/
|-- fake_news_detection/
|-- streamlit_app.py
|-- train_model.py
|-- test_model.py
|-- requirements.txt
```

## How To Run
1. Install dependencies:
   `py -3 -m pip install -r requirements.txt`
2. Train the model:
   `py -3 train_model.py`
3. Run a quick prediction test:
   `py -3 test_model.py`
4. Launch the Streamlit app:
   `py -3 -m streamlit run streamlit_app.py`

## Deployment Notes
- The app uses `@st.cache_resource` so the model is loaded only once per session.
- If `artifacts/model.joblib` does not exist, train the model before deployment.
- For Streamlit Community Cloud, keep `requirements.txt` in the repository root.

## Capstone Alignment
- Clear problem statement and user-focused objective
- Working end-to-end product, not just a notebook
- Deployment through Streamlit
- Reproducible training pipeline and documented setup

## Suggested Documentation PDF Sections
- Title and student details
- Problem statement
- Solution overview and features
- Model pipeline and screenshots
- Tech stack
- Unique points
- Future improvements
