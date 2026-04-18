import unittest

from fake_news_detection.modeling import build_pipeline, combine_text, predict_news


class PipelineTests(unittest.TestCase):
    def test_combine_text_merges_title_and_body(self):
        combined = combine_text("Hello World", "This is the article body.")
        self.assertIn("Hello World", combined)
        self.assertIn("This is the article body.", combined)

    def test_pipeline_can_fit_small_sample(self):
        model = build_pipeline(min_df=1, max_df=1.0)
        X = [
            combine_text("Trusted report", "Officials released audited findings."),
            combine_text("Viral rumor", "Anonymous users claim impossible events happened."),
            combine_text("Verified briefing", "Named agencies published the evidence."),
            combine_text("Conspiracy post", "The article offers no source and no proof."),
        ]
        y = ["REAL", "FAKE", "REAL", "FAKE"]
        model.fit(X, y)
        result = predict_news(
            model,
            title="Breaking rumor",
            text="The claim gives no named source and no supporting evidence.",
        )
        self.assertIn(result["label"], {"REAL", "FAKE"})


if __name__ == "__main__":
    unittest.main()
