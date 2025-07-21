import streamlit as st
import pandas as pd

from marble_interface import (
    new_marble_system,
    train_marble_system,
    infer_marble_system,
    set_dreaming,
    set_autograd,
)


def load_examples(file) -> list[tuple[float, float]]:
    """Load training examples from a CSV file object."""
    df = pd.read_csv(file)
    return list(zip(df["input"].astype(float), df["target"].astype(float)))


def initialize_marble(cfg_path: str):
    """Create a MARBLE system using a YAML configuration path."""
    return new_marble_system(cfg_path)


def run_playground() -> None:
    """Launch the Streamlit MARBLE playground."""
    st.set_page_config(page_title="MARBLE Playground")
    st.title("MARBLE Playground")

    if "marble" not in st.session_state:
        st.session_state["marble"] = None

    cfg_path = st.sidebar.text_input("Config YAML", "config.yaml")
    if st.sidebar.button("Initialize MARBLE"):
        st.session_state["marble"] = initialize_marble(cfg_path)
        st.sidebar.success("System initialized")

    marble = st.session_state.get("marble")
    if marble is None:
        st.info("Initialize MARBLE to begin.")
        return

    dreaming = st.sidebar.checkbox("Dreaming", value=marble.get_brain().dreaming_active)
    set_dreaming(marble, dreaming)

    autograd = st.sidebar.checkbox("Autograd", value=marble.get_autograd_layer() is not None)
    set_autograd(marble, autograd)

    train_file = st.sidebar.file_uploader("Training CSV", type="csv")
    epochs = st.sidebar.number_input("Epochs", min_value=1, value=1, step=1)
    if st.sidebar.button("Train") and train_file is not None:
        examples = load_examples(train_file)
        train_marble_system(marble, examples, epochs=epochs)
        st.sidebar.success("Training complete")
        if marble.get_metrics_visualizer().fig:
            st.pyplot(marble.get_metrics_visualizer().fig)

    st.header("Inference")
    val = st.number_input("Input Value", value=0.0, format="%f")
    if st.button("Infer"):
        out = infer_marble_system(marble, float(val))
        st.write(f"Output: {out}")


if __name__ == "__main__":
    run_playground()
