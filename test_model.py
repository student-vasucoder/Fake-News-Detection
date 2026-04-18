from fake_news_detection.modeling import MODEL_PATH, load_model, predict_news


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model artifact missing. Run `py -3 train_model.py` before testing."
        )

    model = load_model()
    result = predict_news(
        model,
        title="Viral post claims a hidden diplomatic treaty caused the talks to collapse",
        text=(
            "A fast-moving article on social media says the meeting ended after secret "
            "conditions were revealed, but it does not cite any official statement or "
            "named reporting source."
        ),
    )
    print("Prediction:", result["label"])
    print("Confidence:", result["confidence"])


if __name__ == "__main__":
    main()
