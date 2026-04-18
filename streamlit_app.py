from __future__ import annotations

import streamlit as st

from fake_news_detection.modeling import METRICS_PATH, MODEL_PATH, load_model, predict_news
from fake_news_detection.project_info import (
    PROBLEM_STATEMENT,
    PROJECT_TITLE,
    SUCCESS_CRITERIA,
    UNIQUE_POINTS,
)


st.set_page_config(
    page_title=PROJECT_TITLE,
    page_icon="📰",
    layout="wide",
)


@st.cache_resource
def get_model():
    return load_model()


def render_sidebar() -> None:
    with st.sidebar:
        st.title("Project Overview")
        st.write(PROBLEM_STATEMENT)
        st.subheader("Success Criteria")
        for item in SUCCESS_CRITERIA:
            st.write(f"- {item}")

        st.subheader("Highlights")
        for item in UNIQUE_POINTS:
            st.write(f"- {item}")

        if METRICS_PATH.exists():
            st.subheader("Latest Metrics")
            st.code(METRICS_PATH.read_text(encoding="utf-8"), language="text")


def main() -> None:
    render_sidebar()

    st.title(PROJECT_TITLE)
    st.caption("Capstone-ready Streamlit deployment for misinformation screening.")

    if not MODEL_PATH.exists():
        st.error(
            "Model artifact not found. Run `py -3 train_model.py` first to generate "
            "`artifacts/model.joblib`."
        )
        st.stop()

    model = get_model()

    title = st.text_input(
        "Article title",
        value=st.session_state.get("example_title", ""),
        placeholder="Enter the headline of the article...",
    )
    text = st.text_area(
        "Article text",
        value=st.session_state.get("example_text", ""),
        height=280,
        placeholder="Paste the article content here...",
    )

    action_col, example_col = st.columns([2, 1])
    with action_col:
        run_prediction = st.button("Analyze Article", type="primary", use_container_width=True)
    with example_col:
        if st.button("Load Example Input", use_container_width=True):
            st.session_state["example_title"] = (
                "Breaking report claims a miracle cure was hidden from the public"
            )
            st.session_state["example_text"] = (
                "A viral social media post claims that a secret medicine can cure every "
                "major disease overnight. The story cites no verified doctors, no peer-"
                "reviewed study, and no named hospital."
            )
            st.rerun()

    st.info(
        "This tool is an ML screening assistant. It should support human judgment, "
        "not replace professional fact-checking."
    )

    if run_prediction:
        if not title.strip() and not text.strip():
            st.warning("Enter a title, article text, or both before running analysis.")
            st.stop()

        result = predict_news(model, title=title, text=text)
        label = result["label"]
        confidence = result["confidence"]
        probabilities = result["probabilities"] or {}

        if label == "FAKE":
            st.error(f"Prediction: {label}")
        else:
            st.success(f"Prediction: {label}")

        if confidence is not None:
            st.metric("Confidence", f"{confidence * 100:.2f}%")

        if probabilities:
            st.subheader("Class Probabilities")
            st.json(probabilities)

        st.subheader("Recommended Interpretation")
        st.write(
            "Treat this prediction as an initial credibility signal. Verify suspicious "
            "stories with trusted publishers, public records, or dedicated fact-checkers."
        )


if __name__ == "__main__":
    main()

