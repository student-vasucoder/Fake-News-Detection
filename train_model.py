from fake_news_detection.modeling import METRICS_PATH, MODEL_PATH, save_artifacts, train_model


def main() -> None:
    model, result = train_model()
    save_artifacts(model, result)

    print(f"Accuracy: {result.accuracy:.4f}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(result.report)


if __name__ == "__main__":
    main()
