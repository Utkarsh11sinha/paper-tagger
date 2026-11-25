"""Shared paths, constants and human-readable names for arXiv tags."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "raw" / "arxiv_papers.jsonl"
PROCESSED_DIR = DATA_DIR / "processed"
LABELS_PATH = PROCESSED_DIR / "labels.json"

MODELS_DIR = ROOT / "models"
ML_MODEL_PATH = MODELS_DIR / "ml" / "tfidf_logreg.joblib"
BERT_DIR = MODELS_DIR / "bert"

REPORTS_DIR = ROOT / "reports"

SEED = 42

# Categories queried from the arXiv API. Papers are cross-listed, so the final
# tag set also picks up other categories that appear often enough.
DEFAULT_CATEGORIES = [
    "cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.RO", "cs.CR", "cs.DB", "cs.DC",
    "cs.IR", "cs.NE", "cs.SE", "cs.HC", "cs.SI", "cs.GT", "cs.DS", "cs.NI",
    "stat.ML", "stat.ME", "math.OC", "math.ST", "eess.IV", "eess.SP",
    "eess.AS", "eess.SY", "q-bio.QM", "q-fin.CP", "physics.comp-ph",
    "quant-ph", "econ.EM", "astro-ph.IM",
]

CATEGORY_NAMES = {
    "cs.AI": "Artificial Intelligence",
    "cs.AR": "Hardware Architecture",
    "cs.CC": "Computational Complexity",
    "cs.CE": "Computational Engineering",
    "cs.CG": "Computational Geometry",
    "cs.CL": "Computation and Language (NLP)",
    "cs.CR": "Cryptography and Security",
    "cs.CV": "Computer Vision",
    "cs.CY": "Computers and Society",
    "cs.DB": "Databases",
    "cs.DC": "Distributed and Parallel Computing",
    "cs.DL": "Digital Libraries",
    "cs.DM": "Discrete Mathematics",
    "cs.DS": "Data Structures and Algorithms",
    "cs.ET": "Emerging Technologies",
    "cs.FL": "Formal Languages",
    "cs.GR": "Graphics",
    "cs.GT": "Game Theory",
    "cs.HC": "Human-Computer Interaction",
    "cs.IR": "Information Retrieval",
    "cs.IT": "Information Theory",
    "cs.LG": "Machine Learning",
    "cs.LO": "Logic in Computer Science",
    "cs.MA": "Multiagent Systems",
    "cs.MM": "Multimedia",
    "cs.NE": "Neural and Evolutionary Computing",
    "cs.NI": "Networking and Internet Architecture",
    "cs.PF": "Performance",
    "cs.PL": "Programming Languages",
    "cs.RO": "Robotics",
    "cs.SD": "Sound",
    "cs.SE": "Software Engineering",
    "cs.SI": "Social and Information Networks",
    "cs.SY": "Systems and Control",
    "econ.EM": "Econometrics",
    "econ.GN": "General Economics",
    "eess.AS": "Audio and Speech Processing",
    "eess.IV": "Image and Video Processing",
    "eess.SP": "Signal Processing",
    "eess.SY": "Systems and Control (EESS)",
    "math.IT": "Information Theory (Math)",
    "math.NA": "Numerical Analysis",
    "math.OC": "Optimization and Control",
    "math.PR": "Probability",
    "math.ST": "Statistics Theory",
    "physics.comp-ph": "Computational Physics",
    "physics.data-an": "Data Analysis and Statistics",
    "q-bio.NC": "Neurons and Cognition",
    "q-bio.QM": "Quantitative Methods (Biology)",
    "q-fin.CP": "Computational Finance",
    "q-fin.ST": "Statistical Finance",
    "quant-ph": "Quantum Physics",
    "stat.AP": "Statistics Applications",
    "stat.CO": "Statistical Computation",
    "stat.ME": "Statistical Methodology",
    "stat.ML": "Machine Learning (Statistics)",
    "stat.TH": "Statistics Theory",
    "astro-ph.IM": "Astronomy Instrumentation and Methods",
    "cs.NA": "Numerical Analysis (CS)",
}


def tag_name(tag: str) -> str:
    return CATEGORY_NAMES.get(tag, tag)
