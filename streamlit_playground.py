import os
import json
import wave
from io import BytesIO

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from zipfile import ZipFile

import inspect
import marble_interface

from marble_interface import (
    new_marble_system,
    train_marble_system,
    infer_marble_system,
    set_dreaming,
    set_autograd,
)


def _load_image(file_obj: BytesIO) -> np.ndarray:
    img = Image.open(file_obj).convert("RGB")
    return np.array(img, dtype=np.float32)


def _load_audio(file_obj: BytesIO) -> np.ndarray:
    with wave.open(file_obj) as w:
        frames = w.readframes(w.getnframes())
        arr = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    if len(arr) > 0:
        arr /= np.abs(arr).max()
    return arr


def _parse_value(val, zipf: ZipFile | None = None):
    if isinstance(val, (float, int)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            lower = val.lower()
            if zipf is not None and val in zipf.namelist():
                with zipf.open(val) as f:
                    data = f.read()
                    bio = BytesIO(data)
                    if lower.endswith((".png", ".jpg", ".jpeg", ".bmp")):
                        return _load_image(bio)
                    if lower.endswith(".wav"):
                        return _load_audio(bio)
                    return np.frombuffer(data, dtype=np.uint8)
            if os.path.isfile(val):
                with open(val, "rb") as f:
                    data = f.read()
                    bio = BytesIO(data)
                    if lower.endswith((".png", ".jpg", ".jpeg", ".bmp")):
                        return _load_image(bio)
                    if lower.endswith(".wav"):
                        return _load_audio(bio)
                    try:
                        return float(data.decode("utf-8"))
                    except Exception:
                        return np.frombuffer(data, dtype=np.uint8)
            return float(np.mean([ord(c) for c in val]))
    return val


def _pairs_from_df(df: pd.DataFrame, zipf: ZipFile | None = None) -> list[tuple]:
    examples = []
    for _, row in df.iterrows():
        inp = _parse_value(row["input"], zipf)
        tgt = _parse_value(row["target"], zipf)
        examples.append((inp, tgt))
    return examples


def load_examples(file) -> list[tuple]:
    """Load training examples from various dataset formats."""
    name = getattr(file, "name", "")
    ext = os.path.splitext(name)[1].lower()
    if ext == ".csv" or not ext:
        df = pd.read_csv(file)
        return _pairs_from_df(df)
    if ext == ".json" or ext == ".jsonl":
        js = json.load(file)
        df = pd.DataFrame(js)
        return _pairs_from_df(df)
    if ext == ".zip":
        with ZipFile(file) as zf:
            if "dataset.csv" in zf.namelist():
                with zf.open("dataset.csv") as f:
                    df = pd.read_csv(f)
                return _pairs_from_df(df, zf)
            if "dataset.json" in zf.namelist():
                with zf.open("dataset.json") as f:
                    js = json.load(f)
                df = pd.DataFrame(js)
                return _pairs_from_df(df, zf)
            inputs = sorted([n for n in zf.namelist() if n.startswith("inputs/")])
            targets = sorted([n for n in zf.namelist() if n.startswith("targets/")])
            examples = []
            for inp, tgt in zip(inputs, targets):
                examples.append((_parse_value(inp, zf), _parse_value(tgt, zf)))
            return examples
    raise ValueError("Unsupported dataset format")




def initialize_marble(cfg_path: str):
    """Create a MARBLE system using a YAML configuration path."""
    return new_marble_system(cfg_path)


def list_marble_functions() -> list[str]:
    """Return a sorted list of public functions in :mod:`marble_interface`."""
    funcs = []
    for name, obj in inspect.getmembers(marble_interface, inspect.isfunction):
        if not name.startswith("_"):
            funcs.append(name)
    return sorted(funcs)


def execute_marble_function(func_name: str, marble=None, **params):
    """Execute a ``marble_interface`` function by name."""
    if not hasattr(marble_interface, func_name):
        raise ValueError(f"Unknown function: {func_name}")
    func = getattr(marble_interface, func_name)
    sig = inspect.signature(func)
    kwargs = {}
    for name, p in sig.parameters.items():
        if name == "marble" and marble is not None:
            kwargs[name] = marble
            continue
        if name in params:
            kwargs[name] = params[name]
        elif p.default is not inspect.Parameter.empty:
            kwargs[name] = p.default
        else:
            raise ValueError(f"Missing parameter: {name}")
    return func(**kwargs)


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

    mode = st.sidebar.radio("Mode", ["Basic", "Advanced"], key="mode")

    dreaming = st.sidebar.checkbox("Dreaming", value=marble.get_brain().dreaming_active)
    set_dreaming(marble, dreaming)

    autograd = st.sidebar.checkbox("Autograd", value=marble.get_autograd_layer() is not None)
    set_autograd(marble, autograd)

    train_file = st.sidebar.file_uploader(
        "Training Dataset", type=["csv", "json", "jsonl", "zip"]
    )
    epochs = st.sidebar.number_input("Epochs", min_value=1, value=1, step=1)
    if st.sidebar.button("Train") and train_file is not None:
        examples = load_examples(train_file)
        train_marble_system(marble, examples, epochs=epochs)
        st.sidebar.success("Training complete")
        if marble.get_metrics_visualizer().fig:
            st.pyplot(marble.get_metrics_visualizer().fig)

    if mode == "Basic":
        st.header("Inference")
        num_val = st.number_input("Numeric Input", value=0.0, format="%f")
        text_val = st.text_input("Text Input")
        img_file = st.file_uploader("Image Input", type=["png", "jpg", "jpeg", "bmp"], key="img")
        audio_file = st.file_uploader("Audio Input", type=["wav"], key="aud")
        if st.button("Infer"):
            if img_file is not None:
                input_value = _parse_value(img_file.name, ZipFile(BytesIO(img_file.read()), "r"))
            elif audio_file is not None:
                input_value = _load_audio(BytesIO(audio_file.read()))
            elif text_val:
                input_value = _parse_value(text_val)
            else:
                input_value = float(num_val)
            out = infer_marble_system(marble, input_value)
            st.write(f"Output: {out}")
    else:
        st.header("Advanced Function Execution")
        funcs = list_marble_functions()
        selected = st.selectbox("Function", funcs)
        func_obj = getattr(marble_interface, selected)
        sig = inspect.signature(func_obj)
        inputs = {}
        for name, param in sig.parameters.items():
            if name == "marble":
                continue
            default = "" if param.default is inspect.Parameter.empty else str(param.default)
            inputs[name] = st.text_input(name, value=default)
        if st.button("Execute"):
            parsed = {}
            for k, v in inputs.items():
                if v != "":
                    try:
                        parsed[k] = json.loads(v)
                    except Exception:
                        parsed[k] = _parse_value(v)
            result = execute_marble_function(selected, marble, **parsed)
            st.write(result)


if __name__ == "__main__":
    run_playground()
