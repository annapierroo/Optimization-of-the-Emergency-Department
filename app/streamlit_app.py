"""Streamlit UI for encounter-duration prediction."""
import streamlit as st
import sys
from pathlib import Path
from PIL import Image

# Ensure we can import from the same directory
try:
    import backend
except ImportError:
    # Fallback if running from root
    from app import backend

# Define path for the process graph image (Anna's work)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRAPH_PATH = PROJECT_ROOT / "reports" / "figures" / "patient_journey_dfg.png"


def load_model(model_dir=None):
    """
    Loads the latest trained model.
    Inputs: `model_dir` path to directory with serialized model artifacts.
    Outputs: `model` loaded model object.
    """
    # We delegate the heavy lifting to the backend logic
    return backend.load_latest_model(model_dir)


def predict_duration(model, procedures):
    """
    Predicts encounter duration from procedure codes.
    Inputs: `model` trained model instance and `procedures` list of procedure identifiers.
    Outputs: `prediction_minutes` numeric duration estimate.
    """
    # 1. Check if model has feature names (standard in Scikit-Learn)
    if hasattr(model, "feature_names_in_"):
        feature_cols = model.feature_names_in_
    else:
        raise ValueError("The loaded model does not contain feature names. Please retrain.")

    # 2. Preprocess the input list into a dataframe row
    feature_row = backend.preprocess_procedures(procedures, feature_cols)

    # 3. Get the prediction from the backend
    prediction_minutes = backend.predict_duration(model, feature_row)
    
    return prediction_minutes


def render_sidebar():
    """
    Renders input controls.
    Inputs: none needed.
    Outputs: `procedures` list of selected procedures.
    """
    st.sidebar.header("Patient Configuration")
    st.sidebar.markdown("Simulate a patient journey:")

    # Mock list of procedures (In a real app, this would come from config.py or the dataset)
    available_procedures = [
        "Triage", 
        "X-Ray", 
        "Blood Test", 
        "Consultation", 
        "MRI Scan", 
        "Bandaging",
        "Ultrasound",
        "CT Scan"
    ]
    
    procedures = st.sidebar.multiselect(
        "Select Procedures Performed",
        options=available_procedures,
        default=["Triage"]
    )
    
    return procedures


def render_main(prediction_minutes):
    """
    Render prediction results and model metadata.
    Inputs `prediction_minutes` numeric duration estimate.
    Outputs: none.
    """
    col1, col2 = st.columns([1, 1])

    # LEFT COLUMN: Prediction Result
    with col1:
        st.subheader("⏱️ Prediction Result")
        
        if prediction_minutes is not None:
            st.success("Calculation Successful")
            st.metric(
                label="Estimated Discharge Time", 
                value=f"{prediction_minutes} min",
                delta="Based on Random Forest v1"
            )
            st.caption("This prediction is based on the historical duration of similar pathways.")
        else:
            st.info("👈 Please select procedures in the sidebar and click 'Predict'.")

    # RIGHT COLUMN: Process Mining Context (The graph)
    with col2:
        st.subheader("📍 Pathway Visualization")
        if GRAPH_PATH.exists():
            image = Image.open(GRAPH_PATH)
            st.image(image, caption="Hospital Process Flow (DFG)", use_column_width=True)
        else:
            st.warning("⚠️ Process Map not found. Please run 'src/process_discovery.py' first.")


def main():
    """Streamlit entrypoint coordinating inputs and prediction."""
    
    # Page Setup
    st.set_page_config(page_title="ER Optimizer", page_icon="🏥", layout="wide")
    st.title("🏥 Emergency Department Optimization")
    st.markdown("**MLOps Project Exam** | *Predictive Monitoring System*")

    # 1. Render Sidebar to get User Input
    selected_procedures = render_sidebar()

    # 2. Button Logic
    if st.sidebar.button("🔮 Predict Duration"):
        try:
            with st.spinner('Loading model and computing...'):
                # Load the model using our function
                model = load_model()
                
                # specific logic to predict
                minutes = predict_duration(model, selected_procedures)
                
                # Render results
                render_main(minutes)
                
        except Exception as e:
            st.error(f"System Error: {e}")
            st.markdown("💡 *Hint: Have you run `python src/training.py` to train the model yet?*")
    else:
        # Render main area empty/initial state
        render_main(None)


if __name__ == "__main__":
    main()
