import os
import json
import wave
import tempfile
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
    save_marble_system,
    load_marble_system,
    export_core_to_json,
    import_core_from_json,
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


def initialize_marble(cfg_path: str | None = None, yaml_text: str | None = None):
    """Create a MARBLE system from a YAML path or inline YAML text."""
    if yaml_text is not None:
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml") as tmp:
            tmp.write(yaml_text)
            cfg_path = tmp.name
    if cfg_path is None:
        raise ValueError("Either cfg_path or yaml_text must be provided")
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

    cfg_path = st.sidebar.text_input("Config YAML Path", "config.yaml")
    cfg_upload = st.sidebar.file_uploader("Upload YAML", type=["yaml", "yml"], key="cfg_file")
    cfg_text = st.sidebar.text_area("Or paste YAML", key="cfg_text")
    if st.sidebar.button("Initialize MARBLE"):
        yaml_data = None
        if cfg_upload is not None:
            yaml_data = cfg_upload.getvalue().decode("utf-8")
        elif cfg_text.strip():
            yaml_data = cfg_text
        st.session_state["marble"] = initialize_marble(cfg_path if not yaml_data else None, yaml_text=yaml_data)
        st.sidebar.success("System initialized")

    save_path = st.sidebar.text_input("Save Path", "marble.pkl")
    if st.sidebar.button("Save MARBLE") and st.session_state.get("marble") is not None:
        save_marble_system(st.session_state["marble"], save_path)
        st.sidebar.success("Model saved")

    load_file = st.sidebar.file_uploader("Load MARBLE", type=["pkl"], key="load_marble")
    if st.sidebar.button("Load MARBLE") and load_file is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(load_file.read())
        tmp.close()
        st.session_state["marble"] = load_marble_system(tmp.name)
        st.sidebar.success("Model loaded")

    if (
        st.sidebar.button("Export Core JSON")
        and st.session_state.get("marble") is not None
    ):
        js = export_core_to_json(st.session_state["marble"])
        st.sidebar.download_button("Download core.json", js, file_name="core.json")

    core_file = st.sidebar.file_uploader(
        "Import Core JSON", type=["json"], key="load_core"
    )
    if st.sidebar.button("Load Core JSON") and core_file is not None:
        js = core_file.getvalue().decode("utf-8")
        st.session_state["marble"] = import_core_from_json(js)
        st.sidebar.success("Core loaded")

    marble = st.session_state.get("marble")
    if marble is None:
        st.info("Initialize MARBLE to begin.")
        return

    mode = st.sidebar.radio("Mode", ["Basic", "Advanced"], key="mode")

    dreaming = st.sidebar.checkbox("Dreaming", value=marble.get_brain().dreaming_active)
    set_dreaming(marble, dreaming)

    autograd = st.sidebar.checkbox(
        "Autograd", value=marble.get_autograd_layer() is not None
    )
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
        img_file = st.file_uploader(
            "Image Input", type=["png", "jpg", "jpeg", "bmp"], key="img"
        )
        audio_file = st.file_uploader("Audio Input", type=["wav"], key="aud")
        if st.button("Infer"):
            if img_file is not None:
                input_value = _parse_value(
                    img_file.name, ZipFile(BytesIO(img_file.read()), "r")
                )
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
        doc = inspect.getdoc(func_obj) or ""
        if doc:
            st.markdown(doc)
        sig = inspect.signature(func_obj)
        inputs = {}
        for name, param in sig.parameters.items():
            if name == "marble":
                continue
            default = (
                None if param.default is inspect.Parameter.empty else param.default
            )
            ann = param.annotation
            widget = None
            if ann is bool or isinstance(default, bool):
                widget = st.checkbox(
                    name, value=bool(default) if default is not None else False
                )
            elif ann in (int, float) or isinstance(default, (int, float)):
                val = 0 if default is None else float(default)
                widget = st.number_input(name, value=val)
            else:
                widget = st.text_input(
                    name, value="" if default is None else str(default)
                )
            inputs[name] = widget
        if st.button("Execute"):
            parsed = {}
            for k, v in inputs.items():
                if isinstance(v, str) and v != "":
                    try:
                        parsed[k] = json.loads(v)
                    except Exception:
                        parsed[k] = _parse_value(v)
                else:
                    parsed[k] = v
            result = execute_marble_function(selected, marble, **parsed)
            st.write(result)


if __name__ == "__main__":
    run_playground()
