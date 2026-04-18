PROJECT_TITLE = "Fake News Detection Assistant"
PROBLEM_STATEMENT = (
    "Journalists, students, and everyday readers are exposed to large volumes of "
    "online content that can include misleading or fabricated stories. This project "
    "classifies article text as likely REAL or FAKE so users can triage suspicious "
    "content faster."
)
SUCCESS_CRITERIA = [
    "Train a reproducible text-classification model on the provided dataset.",
    "Expose predictions through a simple Streamlit web interface.",
    "Show transparent confidence output so the result is not a black box.",
    "Package the project so it can be retrained and redeployed easily.",
]
UNIQUE_POINTS = [
    "Uses both article title and body text instead of only one field.",
    "Shares one prediction pipeline across training, testing, and the UI.",
    "Includes deployment-ready Streamlit configuration and reproducible artifacts.",
]

