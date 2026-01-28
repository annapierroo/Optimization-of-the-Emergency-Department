"""Streamlit UI for encounter-duration prediction."""


def load_model(model_dir):
    """
    Loads the latest trained model.
    Inputs: `model_dir` path to directory with serialized model artifacts.
    Outputs: `model` loaded model object.
    """
    raise NotImplementedError


def predict_duration(model, procedures):
    """
    Predicts encounter duration from procedure codes.
    Inputs: `model` trained model instance and `procedures` list of procedure identifiers.
    Outputs: `prediction_minutes` numeric duration estimate.
    """
    raise NotImplementedError


def render_sidebar():
    """
    Renders input controls.
    Inputs: none needed.
    Outputs: `procedures` list of selected procedures.
    """
    raise NotImplementedError


def render_main(prediction_minutes):
    """
    Render prediction results and model metadata.
    Inputs `prediction_minutes` numeric duration estimate.
    Outputs: none.
    """
    raise NotImplementedError


def main():
    """Streamlit entrypoint coordinating inputs and prediction."""
    raise NotImplementedError

