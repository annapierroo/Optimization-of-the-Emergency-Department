"""Backend utilities powering Streamlit predictions."""

def load_latest_model(model_dir):
    """
    Loads the most recent model artifact.
    Inputs: `model_dir` path to model artifact directory.
    Outputs: `model` deserialized model object.
    """
    raise NotImplementedError


def preprocess_procedures(procedures):
    """
    Converts raw procedure list into model-ready features.
    Inputs: `procedures` list of procedure identifiers.
    Outputs: `feature_row` single-row feature vector for inference.
    """
    raise NotImplementedError


def predict_duration(model, feature_row):
    """
    Runs model inference.
    Inputs `model` trained model object, `feature_row` model-ready features.
    Outputs: `prediction_minutes` numeric duration prediction.
    """
    raise NotImplementedError

