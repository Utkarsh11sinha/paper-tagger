"""Load whichever of the two models are trained (used by app.py and suggest_tags.py)."""
from dl_model import predict as dl_predict
from ml_model import predict as ml_predict


def load_available():
    """Return {"ml": MLTagger, "bert": BertTagger} for the models found on disk."""
    taggers = {}
    if ml_predict.is_trained():
        taggers["ml"] = ml_predict.MLTagger()
    if dl_predict.is_trained():
        taggers["bert"] = dl_predict.BertTagger()
    return taggers
