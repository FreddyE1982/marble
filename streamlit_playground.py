import os
import json
import wave
import tempfile
from io import BytesIO
import io
import runpy
import contextlib

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from zipfile import ZipFile

import networkx as nx
import plotly.graph_objs as go

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
    load_hf_dataset,
)

from metrics_dashboard import MetricsDashboard
import pkgutil
import importlib
import yaml
import torch
from transformers import AutoModel


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


def load_hf_examples(
    dataset_name: str,
    split: str,
    input_key: str = "input",
    target_key: str = "target",
    limit: int | None = None,
) -> list[tuple]:
    """Load ``(input, target)`` pairs from a Hugging Face dataset."""
    return load_hf_dataset(dataset_name, split, input_key, target_key, limit)


def load_hf_model(model_name: str):
    """Return a pretrained model from the Hugging Face Hub."""
    return AutoModel.from_pretrained(model_name, trust_remote_code=True)


def model_summary(model: torch.nn.Module) -> str:
    """Return a textual summary of ``model`` parameters."""
    lines = ["Layer | Shape | Params"]
    total = 0
    for name, param in model.named_parameters():
        lines.append(f"{name} | {tuple(param.shape)} | {param.numel()}")
        total += param.numel()
    lines.append(f"Total parameters: {total}")
    return "\n".join(lines)


def convert_hf_model(
    model_name: str,
    core_params: dict | None = None,
    nb_params: dict | None = None,
    brain_params: dict | None = None,
    dataloader_params: dict | None = None,
) -> marble_interface.MARBLE:
    """Convert ``model_name`` from HF Hub into a MARBLE system."""
    model = load_hf_model(model_name)
    return marble_interface.convert_pytorch_model(
        model,
        core_params=core_params,
        nb_params=nb_params,
        brain_params=brain_params,
        dataloader_params=dataloader_params,
    )


def preview_file_dataset(file, limit: int = 5) -> pd.DataFrame:
    """Return a DataFrame preview of a local dataset file."""
    if hasattr(file, "getvalue"):
        data = file.getvalue()
    else:
        data = file.read()
    buf = BytesIO(data)
    df = pd.DataFrame(load_examples(buf)[:limit], columns=["input", "target"])
    return df


def preview_hf_dataset(
    dataset_name: str,
    split: str,
    input_key: str = "input",
    target_key: str = "target",
    limit: int = 5,
) -> pd.DataFrame:
    """Return a DataFrame preview of a Hugging Face dataset."""
    ex = load_hf_examples(dataset_name, split, input_key, target_key, limit=limit)
    return pd.DataFrame(ex, columns=["input", "target"])


def start_metrics_dashboard(
    marble,
    host: str = "localhost",
    port: int = 8050,
    update_interval: int = 1000,
    window_size: int = 10,
) -> MetricsDashboard:
    """Start a live metrics dashboard for ``marble`` and return it."""
    dash = MetricsDashboard(
        marble.get_metrics_visualizer(),
        host=host,
        port=port,
        update_interval=update_interval,
        window_size=window_size,
    )
    dash.start()
    return dash


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


def list_repo_modules() -> list[str]:
    """Return a sorted list of top-level modules in the repository."""
    root = os.path.dirname(__file__)
    modules = []
    for mod in pkgutil.iter_modules([root]):
        if mod.ispkg:
            continue
        name = mod.name
        if name.startswith("_") or name in {"streamlit_playground", "tests"}:
            continue
        modules.append(name)
    return sorted(modules)


def list_module_functions(module_name: str) -> list[str]:
    """Return a sorted list of public functions in ``module_name``."""
    module = importlib.import_module(module_name)
    funcs = []
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if not name.startswith("_"):
            funcs.append(name)
    return sorted(funcs)


def execute_module_function(
    module_name: str, func_name: str, marble=None, **params
) -> object:
    """Execute ``func_name`` from ``module_name`` with ``params``."""
    module = importlib.import_module(module_name)
    if not hasattr(module, func_name):
        raise ValueError(f"Unknown function: {func_name}")
    func = getattr(module, func_name)
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


def execute_function_sequence(
    steps: list[dict], marble=None
) -> list[object]:
    """Execute a sequence of functions defined in ``steps``.

    Each step is a dictionary with keys:

    ``module`` (optional): module name containing the function
    ``func``: function name
    ``params`` (optional): dictionary of parameters

    If ``module`` is omitted or ``None``, the function is looked up in
    :mod:`marble_interface`. Results from each step are collected and
    returned in order.
    """

    results: list[object] = []
    for step in steps:
        module_name = step.get("module")
        func_name = step["func"]
        params = step.get("params", {})
        if module_name:
            results.append(
                execute_module_function(module_name, func_name, marble, **params)
            )
        else:
            results.append(
                execute_marble_function(func_name, marble, **params)
            )
    return results


def save_pipeline_to_json(pipeline: list[dict], path: str) -> None:
    """Save a function pipeline to ``path`` as JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pipeline, f, indent=2)


def remove_pipeline_step(pipeline: list[dict], index: int) -> list[dict]:
    """Return ``pipeline`` with the step at ``index`` removed."""
    if index < 0 or index >= len(pipeline):
        raise IndexError("index out of range")
    pipeline.pop(index)
    return pipeline


def move_pipeline_step(pipeline: list[dict], old_index: int, new_index: int) -> list[dict]:
    """Move a pipeline step from ``old_index`` to ``new_index`` and return it."""
    if old_index < 0 or old_index >= len(pipeline):
        raise IndexError("old_index out of range")
    if new_index < 0 or new_index >= len(pipeline):
        raise IndexError("new_index out of range")
    step = pipeline.pop(old_index)
    pipeline.insert(new_index, step)
    return pipeline


def load_yaml_manual() -> str:
    """Return the contents of ``yaml-manual.txt``."""
    path = os.path.join(os.path.dirname(__file__), "yaml-manual.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def set_yaml_value(yaml_text: str, path: str, value: object) -> str:
    """Return ``yaml_text`` with ``path`` updated to ``value``.

    Parameters
    ----------
    yaml_text:
        YAML configuration as text.
    path:
        Dot-separated key path specifying where to insert the value.
    value:
        Value to assign. Strings are inserted as-is while other types are
        serialized using ``json`` when possible.
    """

    data = yaml.safe_load(yaml_text) if yaml_text else {}
    cur = data
    keys = path.split(".")
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value
    return yaml.safe_dump(data, sort_keys=False)


def core_to_networkx(core) -> nx.DiGraph:
    """Return a NetworkX graph representing ``core``."""
    g = nx.DiGraph()
    for n in core.neurons:
        g.add_node(n.id, neuron_type=n.neuron_type)
    for s in core.synapses:
        g.add_edge(s.source, s.target, weight=s.weight)
    return g


def core_figure(core, layout: str = "spring") -> go.Figure:
    """Create a Plotly figure visualizing ``core``."""
    g = core_to_networkx(core)
    if layout == "circular":
        pos = nx.circular_layout(g)
    else:
        pos = nx.spring_layout(g, seed=42)
    edge_x, edge_y = [], []
    for u, v in g.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    node_x, node_y, text = [], [], []
    for n in g.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        text.append(f"id={n}<br>{g.nodes[n].get('neuron_type')}")
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
    )
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        text=text,
        marker=dict(size=6, color="#1f77b4"),
    )
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
    return fig


def load_pipeline_from_json(file) -> list[dict]:
    """Load a function pipeline from a JSON file or file-like object."""
    if hasattr(file, "read"):
        data = file.read()
    else:
        with open(file, "r", encoding="utf-8") as f:
            data = f.read()
    return json.loads(data)


def run_custom_code(code: str, marble=None) -> object:
    """Execute arbitrary Python ``code`` with ``marble`` in scope.

    The snippet may assign a variable named ``result`` which will be
    returned from this function. Any printed output will appear in the
    Streamlit interface when used there.
    """

    locals_dict: dict[str, object] = {"marble": marble}
    exec(code, {}, locals_dict)
    return locals_dict.get("result")


def list_example_projects() -> list[str]:
    """Return available example project script names sorted alphabetically."""
    ex_dir = os.path.join(os.path.dirname(__file__), "examples")
    names = [f for f in os.listdir(ex_dir) if f.endswith(".py")]
    return sorted(names)


def load_example_code(project_name: str) -> str:
    """Return the source code of ``project_name`` from the examples directory."""
    ex_dir = os.path.join(os.path.dirname(__file__), "examples")
    path = os.path.join(ex_dir, project_name)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def run_example_project(project_name: str) -> str:
    """Execute an example project script and return captured output."""
    ex_dir = os.path.join(os.path.dirname(__file__), "examples")
    path = os.path.join(ex_dir, project_name)
    out_buf = io.StringIO()
    err_buf = io.StringIO()
    with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
        runpy.run_path(path, run_name="__main__")
    return out_buf.getvalue() + err_buf.getvalue()


def run_playground() -> None:
    """Launch the Streamlit MARBLE playground."""
    st.set_page_config(page_title="MARBLE Playground")
    st.title("MARBLE Playground")

    if "marble" not in st.session_state:
        st.session_state["marble"] = None
    if "dashboard" not in st.session_state:
        st.session_state["dashboard"] = None
    if "hf_examples" not in st.session_state:
        st.session_state["hf_examples"] = []
    if "pipeline" not in st.session_state:
        st.session_state["pipeline"] = []

    cfg_path = st.sidebar.text_input("Config YAML Path", "config.yaml")
    cfg_upload = st.sidebar.file_uploader(
        "Upload YAML", type=["yaml", "yml"], key="cfg_file"
    )
    cfg_text = st.sidebar.text_area("Or paste YAML", key="cfg_text")
    if st.sidebar.button("Initialize MARBLE"):
        yaml_data = None
        if cfg_upload is not None:
            yaml_data = cfg_upload.getvalue().decode("utf-8")
        elif cfg_text.strip():
            yaml_data = cfg_text
        else:
            with open(cfg_path, "r", encoding="utf-8") as f:
                yaml_data = f.read()
        st.session_state["marble"] = initialize_marble(
            cfg_path if not yaml_data else None, yaml_text=yaml_data
        )
        st.session_state["config_yaml"] = yaml_data
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

    if "config_yaml" in st.session_state:
        with st.sidebar.expander("Current Config"):
            st.code(st.session_state["config_yaml"], language="yaml")
    with st.sidebar.expander("YAML Manual"):
        st.code(load_yaml_manual())

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

    dash_toggle = st.sidebar.checkbox("Metrics Dashboard")
    dash_host = st.sidebar.text_input("Dashboard Host", "localhost")
    dash_port = st.sidebar.number_input("Dashboard Port", value=8050, step=1)
    dash_interval = st.sidebar.number_input("Dashboard Interval", value=1000, step=100)
    if dash_toggle and st.session_state.get("dashboard") is None:
        st.session_state["dashboard"] = start_metrics_dashboard(
            marble,
            host=dash_host,
            port=int(dash_port),
            update_interval=int(dash_interval),
        )
        st.sidebar.success("Dashboard started")
    if not dash_toggle and st.session_state.get("dashboard") is not None:
        st.session_state["dashboard"].stop()
        st.session_state["dashboard"] = None

    train_file = st.sidebar.file_uploader(
        "Training Dataset", type=["csv", "json", "jsonl", "zip"]
    )
    if train_file is not None:
        st.sidebar.dataframe(preview_file_dataset(train_file), use_container_width=True)
    hf_name = st.sidebar.text_input("HF Dataset Name")
    hf_split = st.sidebar.text_input("HF Split", "train[:100]")
    hf_input = st.sidebar.text_input("Input Key", "input")
    hf_target = st.sidebar.text_input("Target Key", "target")
    hf_limit = st.sidebar.number_input("HF Limit", min_value=1, value=100, step=1)
    if st.sidebar.button("Load HF Dataset") and hf_name and hf_split:
        st.session_state["hf_examples"] = load_hf_examples(
            hf_name,
            hf_split,
            input_key=hf_input,
            target_key=hf_target,
            limit=int(hf_limit),
        )
        st.sidebar.success(
            f"Loaded {len(st.session_state['hf_examples'])} examples from HF"
        )
    if st.session_state.get("hf_examples"):
        st.sidebar.dataframe(
            pd.DataFrame(
                st.session_state["hf_examples"][:5],
                columns=["input", "target"],
            ),
            use_container_width=True,
        )
    epochs = st.sidebar.number_input("Epochs", min_value=1, value=1, step=1)
    if st.sidebar.button("Train"):
        examples = []
        if train_file is not None:
            examples = load_examples(train_file)
        elif "hf_examples" in st.session_state:
            examples = st.session_state["hf_examples"]
        if not examples:
            st.sidebar.error("No dataset loaded")
        else:
            train_marble_system(marble, examples, epochs=epochs)
            st.sidebar.success("Training complete")
            if marble.get_metrics_visualizer().fig:
                st.pyplot(marble.get_metrics_visualizer().fig)

    eval_file = st.sidebar.file_uploader(
        "Evaluation Dataset", type=["csv", "json", "jsonl", "zip"], key="eval_file"
    )
    if eval_file is not None:
        st.sidebar.dataframe(preview_file_dataset(eval_file), use_container_width=True)
    if st.sidebar.button("Evaluate") and eval_file is not None:
        eval_examples = load_examples(eval_file)
        mse = marble_interface.evaluate_marble_system(marble, eval_examples)
        st.sidebar.write(f"MSE: {mse}")

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
        tab_iface, tab_mod, tab_pipe, tab_code, tab_vis, tab_cfg, tab_model, tab_proj = st.tabs(
            [
                "marble_interface",
                "Modules",
                "Pipeline",
                "Custom Code",
                "Visualization",
                "Config Editor",
                "Model Conversion",
                "Projects",
            ]
        )

        with tab_iface:
            funcs = list_marble_functions()
            selected = st.selectbox("Function", funcs, key="iface_func")
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
            if st.button("Execute", key="iface_execute"):
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

        with tab_mod:
            modules = list_repo_modules()
            module_choice = st.selectbox("Module", modules, key="mod_select")
            funcs = list_module_functions(module_choice)
            func_choice = st.selectbox("Function", funcs, key="mod_func")
            func_obj = getattr(importlib.import_module(module_choice), func_choice)
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
                        name,
                        value=bool(default) if default is not None else False,
                        key=f"{module_choice}_{name}",
                    )
                elif ann in (int, float) or isinstance(default, (int, float)):
                    val = 0 if default is None else float(default)
                    widget = st.number_input(
                        name, value=val, key=f"{module_choice}_{name}"
                    )
                else:
                    widget = st.text_input(
                        name,
                        value="" if default is None else str(default),
                        key=f"{module_choice}_{name}",
                    )
                inputs[name] = widget
            if st.button("Execute", key="mod_execute"):
                parsed = {}
                for k, v in inputs.items():
                    if isinstance(v, str) and v != "":
                        try:
                            parsed[k] = json.loads(v)
                        except Exception:
                            parsed[k] = _parse_value(v)
                    else:
                        parsed[k] = v
                result = execute_module_function(
                    module_choice, func_choice, marble, **parsed
                )
                st.write(result)

        with tab_pipe:
            st.write("Build a sequence of function calls.")
            if st.session_state["pipeline"]:
                for i, step in enumerate(st.session_state["pipeline"]):
                    cols = st.columns(4)
                    cols[0].markdown(
                        f"**{i+1}.** `{step.get('module') or 'marble_interface'}.{step['func']}`"
                    )
                    if cols[1].button("⬆", key=f"pipe_up_{i}") and i > 0:
                        move_pipeline_step(st.session_state["pipeline"], i, i - 1)
                    if cols[2].button("⬇", key=f"pipe_down_{i}") and i < len(st.session_state["pipeline"]) - 1:
                        move_pipeline_step(st.session_state["pipeline"], i, i + 1)
                    if cols[3].button("✕", key=f"pipe_rm_{i}"):
                        remove_pipeline_step(st.session_state["pipeline"], i)
                        st.experimental_rerun()
            with st.expander("Load/Save Pipeline"):
                pipe_up = st.file_uploader("Load Pipeline", type=["json"], key="pipe_load")
                if st.button("Load", key="load_pipe") and pipe_up is not None:
                    st.session_state["pipeline"] = load_pipeline_from_json(pipe_up)
                    st.success("Pipeline loaded")
                if st.session_state["pipeline"]:
                    pipe_json = json.dumps(st.session_state["pipeline"], indent=2)
                    st.download_button("Download pipeline.json", pipe_json, file_name="pipeline.json")
            with st.expander("Add Step"):
                mode_sel = st.radio(
                    "Source", ["marble_interface", "module"], key="pipe_src"
                )
                if mode_sel == "module":
                    mod = st.selectbox("Module", list_repo_modules(), key="pipe_mod")
                    funcs = list_module_functions(mod)
                    func = st.selectbox("Function", funcs, key="pipe_func")
                else:
                    mod = None
                    funcs = list_marble_functions()
                    func = st.selectbox("Function", funcs, key="pipe_iface_func")
                func_obj = (
                    getattr(importlib.import_module(mod), func)
                    if mod
                    else getattr(marble_interface, func)
                )
                sig = inspect.signature(func_obj)
                params = {}
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
                            name,
                            value=bool(default) if default is not None else False,
                            key=f"pipe_{name}",
                        )
                    elif ann in (int, float) or isinstance(default, (int, float)):
                        val = 0 if default is None else float(default)
                        widget = st.number_input(
                            name, value=val, key=f"pipe_{name}"
                        )
                    else:
                        widget = st.text_input(
                            name,
                            value="" if default is None else str(default),
                            key=f"pipe_{name}",
                        )
                    params[name] = widget
                if st.button("Add to Pipeline", key="add_pipe"):
                    parsed = {}
                    for k, v in params.items():
                        if isinstance(v, str) and v != "":
                            try:
                                parsed[k] = json.loads(v)
                            except Exception:
                                parsed[k] = _parse_value(v)
                        else:
                            parsed[k] = v
                    st.session_state["pipeline"].append(
                        {"module": mod, "func": func, "params": parsed}
                    )
            if st.button("Run Pipeline") and st.session_state["pipeline"]:
                res = execute_function_sequence(
                    st.session_state["pipeline"], marble
                )
                for out in res:
                    st.write(out)
            if st.button("Clear Pipeline"):
                st.session_state["pipeline"] = []

        with tab_code:
            st.write("Execute custom Python code. Use the `marble` variable to access the system.")
            code = st.text_area("Code", height=200, key="custom_code")
            if st.button("Run Code", key="run_custom_code") and code.strip():
                out = run_custom_code(code, marble)
                st.write(out)

        with tab_vis:
            st.write("Visualize the current MARBLE core.")
            if st.button("Generate Graph", key="show_graph"):
                fig = core_figure(marble.get_core())
                st.plotly_chart(fig, use_container_width=True)

        with tab_cfg:
            st.write("Edit the active YAML configuration.")
            param = st.text_input("Parameter Path", key="cfg_param")
            val = st.text_input("Value", key="cfg_value")
            if st.button("Update Config", key="cfg_update"):
                if "config_yaml" not in st.session_state:
                    st.error("No configuration loaded")
                else:
                    new_yaml = set_yaml_value(
                        st.session_state["config_yaml"], param, _parse_value(val)
                    )
                    st.session_state["config_yaml"] = new_yaml
                    st.code(new_yaml, language="yaml")
            if st.button("Reinitialize", key="cfg_reinit"):
                if "config_yaml" not in st.session_state:
                    st.error("No configuration loaded")
                else:
                    st.session_state["marble"] = initialize_marble(
                        None, yaml_text=st.session_state["config_yaml"]
                    )
                    st.success("Reinitialized MARBLE")

        with tab_model:
            st.write("Convert a pretrained Hugging Face model into MARBLE.")
            model_name = st.text_input("Model Name", key="hf_model_name")
            if st.button("Preview Model", key="hf_preview") and model_name:
                try:
                    mdl = load_hf_model(model_name)
                    st.session_state["hf_model"] = mdl
                    st.text(model_summary(mdl))
                except Exception as e:
                    st.error(str(e))
            if (
                st.button("Convert to MARBLE", key="hf_convert")
                and model_name
            ):
                try:
                    marble = convert_hf_model(model_name)
                    st.session_state["marble"] = marble
                    st.success("Model converted")
                except Exception as e:
                    st.error(str(e))

        with tab_proj:
            st.write("Run example projects to explore MARBLE's capabilities.")
            projs = list_example_projects()
            proj = st.selectbox("Project Script", projs, key="proj_select")
            with st.expander("Show Code"):
                st.code(load_example_code(proj), language="python")
            if st.button("Run Project", key="proj_run"):
                try:
                    output = run_example_project(proj)
                    st.text(output if output else "Project finished with no output")
                except Exception as e:
                    st.error(str(e))


if __name__ == "__main__":
    run_playground()
