# Fact or Fake: A Fake News Detection Classifying System

Name: Madhav Kumar Das  
Roll Number: 23053632  
Batch/Program: 2023-2027 / B.tech CSE  

## 1. Abstract

The rapid growth of digital media and social platforms has made it easier than ever to publish and share information. While this improves access to news, it also enables the large-scale spread of misleading, manipulated, and completely fabricated articles. Such misinformation can influence public opinion, damage trust, and create confusion during political, social, and health-related events.

This project presents a machine learning based fake news detection system that classifies news content as `REAL` or `FAKE` using the article headline and body text. The project uses a TF-IDF feature extraction pipeline combined with Logistic Regression to learn language patterns associated with reliable and unreliable news articles. The solution has been organized as a reusable Python project with a training script, a testing script, saved model artifacts, and a deployable Streamlit web application for real-time prediction.

The updated version of the project focuses on reproducibility, usability, and deployment readiness. It provides an end-to-end workflow from training to inference and is suitable for capstone demonstration and submission.

## 2. Problem Statement

Fake news is one of the most serious challenges in the modern information ecosystem. Online users consume large volumes of content every day, and many articles are forwarded or shared before their credibility is verified. Manual fact-checking is time-consuming, and ordinary users may not know how to assess whether a news article is trustworthy.

The goal of this project is to build a practical automated system that helps users quickly screen article content and identify whether the news is likely to be real or fake. This system is intended as a decision-support tool for students, readers, researchers, and journalists who need a fast first-pass credibility signal before performing deeper verification.

## 3. Objectives

- Build a reliable machine learning model for classifying fake and real news articles.
- Use both article title and article body to improve classification quality.
- Create a reusable training and inference pipeline instead of isolated scripts.
- Deploy the model through a Streamlit interface for live user interaction.
- Produce a capstone-ready project with documentation, testing, and deployment support.

## 4. Dataset Description

The primary dataset used in this project is:

- `Data/fake_or_real_news.csv`

### Dataset Summary

- Total records: **6,335**
- Labels: **FAKE** and **REAL**
- Class balance:
  - FAKE: **3,164**
  - REAL: **3,171**

### Available Fields

- `title`
- `text`
- `label`

The dataset is close to balanced, which makes it suitable for a supervised binary classification problem. In the improved project pipeline, the title and text are cleaned and combined into a single feature field so that the model can learn from both the headline style and article content.

## 5. Methodology

### 5.1 Data Preparation

The dataset is loaded using Pandas. Missing values in the `title` and `text` columns are handled by replacing them with empty strings. The project then combines both fields into one normalized text input. This step helps the model capture patterns from headlines as well as article body language.

### 5.2 Feature Engineering

The project uses **TF-IDF Vectorization** to convert the combined article text into numeric features. TF-IDF measures how important a word is within a document relative to the full dataset. This is effective for text classification because it highlights discriminative terms and downweights overly common words.

The vectorizer is configured with:

- English stop-word removal
- Unigrams and bigrams
- Document-frequency pruning
- Sublinear term-frequency scaling

These settings improve generalization and reduce noise in the feature space.

### 5.3 Model Selection

The selected classifier is **Logistic Regression**, which is a strong baseline for high-dimensional sparse text data. It is efficient, interpretable, and performs well on binary classification tasks such as fake-news detection.

### 5.4 Training Strategy

The data is split into training and testing sets using an 80:20 ratio with stratification to preserve class balance. A reproducible random seed is used so the project produces consistent results across runs.

### 5.5 Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

These metrics are written to `artifacts/metrics.txt` for documentation and reuse in the Streamlit interface.

## 6. System Design

The project was refactored from simple scripts into a cleaner product-style structure.

### Project Structure

```text
Fake-News-Detection/
|-- Data/
|-- Report/
|-- .streamlit/config.toml
|-- artifacts/
|-- fake_news_detection/
|   |-- __init__.py
|   |-- modeling.py
|   |-- project_info.py
|-- streamlit_app.py
|-- train_model.py
|-- test_model.py
|-- tests/test_pipeline.py
|-- requirements.txt
|-- README.md
```

### Functional Components

- `train_model.py`
  Trains the pipeline and saves the model artifact and evaluation metrics.

- `test_model.py`
  Runs a smoke test on the trained model using sample news content.

- `fake_news_detection/modeling.py`
  Contains reusable logic for data loading, preprocessing, pipeline creation, training, saving, and prediction.

- `streamlit_app.py`
  Provides the web-based user interface for live article analysis.

- `.streamlit/config.toml`
  Stores deployment-friendly Streamlit configuration.

## 7. Streamlit Deployment

To make the project suitable for capstone demonstration, a deployable Streamlit application was added. The interface allows the user to:

- Enter an article title
- Paste article content
- Run prediction using the trained model
- View the predicted label
- View model confidence
- Review the latest saved training metrics

The app uses `@st.cache_resource` to load the trained model efficiently and avoid repeated initialization during reruns. This is important for smoother deployment and user experience.

### Run Commands

```bash
py -3 -m pip install -r requirements.txt
py -3 train_model.py
py -3 -m streamlit run streamlit_app.py
```

## 8. Experimental Results

After retraining the improved pipeline on the provided dataset, the following results were obtained:

- Accuracy: **0.9337**
- Training samples: **5,068**
- Testing samples: **1,267**

### Confusion Matrix

| Actual \\ Predicted | FAKE | REAL |
|---|---:|---:|
| FAKE | 596 | 37 |
| REAL | 47 | 587 |

### Classification Report

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| FAKE | 0.9269 | 0.9415 | 0.9342 |
| REAL | 0.9407 | 0.9259 | 0.9332 |

These results show that the model performs strongly and remains balanced across both classes, which is important in fake-news detection because both false positives and false negatives can be costly.

## 9. Features of the Final Project

- End-to-end training and prediction pipeline
- Reusable Python package structure
- Combined headline and body-text modeling
- Saved model and metrics artifacts
- Unit tests for core pipeline behavior
- Streamlit-ready user interface
- Clear documentation for training, testing, and deployment

## 10. Unique Points

- The project is no longer limited to a one-time training script; it now follows a reusable software structure.
- The same prediction logic is shared across training, testing, and deployment, reducing inconsistency.
- The interface exposes confidence scores so users understand the strength of each prediction.
- The project is ready for capstone submission because it includes code organization, metrics, documentation, and deployment support.

## 11. Limitations

- The model is trained only on the provided labeled dataset and may not generalize equally well to every real-world news topic.
- It classifies article text but does not independently verify claims using live fact-checking sources.
- The current system does not use source reputation, URL metadata, publisher history, or external knowledge retrieval.
- It should be treated as an assistance tool rather than a final authority on truthfulness.

## 12. Future Improvements

- Add explainability features showing the most influential words for each prediction.
- Incorporate source-domain and URL credibility features.
- Use transformer-based models such as BERT for deeper contextual understanding.
- Add multilingual fake-news detection support.
- Integrate live fact-checking APIs or retrieval-based verification for stronger real-world usage.

## 13. Screenshots

Add the following screenshots before final PDF submission:

- Home page of the Streamlit application
- Article input form
- Prediction output showing label and confidence
- Training metrics or terminal output after model training

## 14. Conclusion

This project addresses the growing problem of misinformation by building a practical fake news detection system using machine learning. The improved version of the project not only delivers good classification performance but also transforms the work into a more complete software product. With structured code, evaluation artifacts, tests, and a Streamlit deployment layer, the project is now better aligned with capstone expectations and is easier to demonstrate, maintain, and extend.

Overall, the project shows how natural language processing and supervised machine learning can be used to support faster and more consistent screening of potentially misleading online content.
