from __future__ import annotations

import contextlib
import glob
import io
import json
import os
import runpy
import sys
import tempfile
import wave
from io import BytesIO

# ruff: noqa: E402


sys.modules.setdefault("streamlit_playground", sys.modules[__name__])
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*google._upb._message.*PyType_Spec.*",
    category=DeprecationWarning,
)

import importlib
import inspect
import pkgutil
import time
from zipfile import ZipFile

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import pytest
import streamlit as st
import streamlit.components.v1 as components
import torch
import yaml
from bit_tensor_dataset import BitTensorDataset
from PIL import Image
from event_bus import PROGRESS_EVENT, ProgressEvent, global_event_bus
from pipeline import Pipeline

from huggingface_utils import (
    hf_load_dataset as load_hf_dataset,
    hf_load_model as load_hf_model,
    hf_login,
)
from huggingface_hub import HfApi
from transformers import AutoModel
from marble_interface import load_hf_dataset as _iface_load_hf_dataset

import marble_interface
from marble_interface import (
    add_neuron_to_marble,
    add_synapse_to_marble,
    decrease_marble_representation,
    expand_marble_core,
    export_core_to_json,
    freeze_synapses_fraction,
    import_core_from_json,
    increase_marble_representation,
    infer_marble_system,
    load_marble_system,
    new_marble_system,
    randomize_core_representations,
    reset_core_representations,
    run_core_message_passing,
    save_marble_system,
    set_autograd,
    set_dreaming,
    train_autoencoder,
    train_marble_system,
)
from bit_tensor_dataset import (
    object_to_bytes,
    bytes_to_tensors,
    flatten_tensor_to_bitstream,
)
from marble_registry import MarbleRegistry
from metrics_dashboard import MetricsDashboard


def _detect_device() -> str:
    """Detect whether the UI is viewed on desktop or mobile."""
    if "device" in st.session_state:
        return st.session_state["device"]
    params = st.experimental_get_query_params()
    if "device" in params:
        st.session_state["device"] = params["device"][0]
        return st.session_state["device"]
    components.html(
        """
        <script>
        const device = window.innerWidth < 768 ? 'mobile' : 'desktop';
        const params = new URLSearchParams(window.location.search);
        if (params.get('device') !== device) {
            params.set('device', device);
            window.location.search = params.toString();
        }
        </script>
        """,
        height=0,
    )
    params = st.experimental_get_query_params()
    st.session_state["device"] = params.get("device", ["desktop"])[0]
    return st.session_state["device"]


def _auto_refresh(interval_ms: int, key: str) -> None:
    """Rerun the app periodically when called inside the main loop."""
    now = time.time()
    last = st.session_state.get(key, 0.0)
    if now - last >= interval_ms / 1000.0:
        st.session_state[key] = now
        st.experimental_rerun()


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


from typing import Any


def _parse_value(val: Any, zipf: ZipFile | None = None) -> float | np.ndarray:
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
    if ext in {".xlsx", ".xls"}:
        df = pd.read_excel(file)
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
            if "dataset.xlsx" in zf.namelist() or "dataset.xls" in zf.namelist():
                fname = (
                    "dataset.xlsx" if "dataset.xlsx" in zf.namelist() else "dataset.xls"
                )
                with zf.open(fname) as f:
                    df = pd.read_excel(f)
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


def load_value_list(file) -> list[float]:
    """Load a list of numeric values from ``file``."""
    name = getattr(file, "name", "")
    ext = os.path.splitext(name)[1].lower()
    if ext == ".csv" or not ext:
        df = pd.read_csv(file)
        col = "value" if "value" in df.columns else df.columns[0]
        return df[col].astype(float).tolist()
    if ext in {".xlsx", ".xls"}:
        df = pd.read_excel(file)
        col = "value" if "value" in df.columns else df.columns[0]
        return df[col].astype(float).tolist()
    if ext in {".json", ".jsonl"}:
        js = json.load(file)
        if isinstance(js, list):
            if js and isinstance(js[0], dict):
                key = "value" if "value" in js[0] else list(js[0].keys())[0]
                return [float(e[key]) for e in js]
            return [float(x) for x in js]
    if ext == ".txt":
        text = file.read().decode("utf-8")
        return [float(v) for v in text.split() if v.strip()]
    if ext == ".zip":
        with ZipFile(file) as zf:
            if "values.csv" in zf.namelist():
                with zf.open("values.csv") as f:
                    df = pd.read_csv(f)
                col = "value" if "value" in df.columns else df.columns[0]
                return df[col].astype(float).tolist()
            if "values.xlsx" in zf.namelist() or "values.xls" in zf.namelist():
                fname = (
                    "values.xlsx" if "values.xlsx" in zf.namelist() else "values.xls"
                )
                with zf.open(fname) as f:
                    df = pd.read_excel(f)
                col = "value" if "value" in df.columns else df.columns[0]
                return df[col].astype(float).tolist()
            if "values.json" in zf.namelist():
                with zf.open("values.json") as f:
                    js = json.load(f)
                if js and isinstance(js[0], dict):
                    key = "value" if "value" in js[0] else list(js[0].keys())[0]
                    return [float(e[key]) for e in js]
                return [float(x) for x in js]
    raise ValueError("Unsupported dataset format")


def load_hf_examples(
    dataset_name: str,
    split: str,
    input_key: str = "input",
    target_key: str = "target",
    limit: int | None = None,
) -> list[tuple]:
    """Load ``(input, target)`` pairs from a Hugging Face dataset."""
    return load_hf_dataset(
        dataset_name, split, input_key, target_key, limit, streaming=False
    )


def search_hf_datasets(query: str, limit: int = 20) -> list[str]:
    """Return dataset IDs from the Hugging Face Hub matching ``query``."""
    from huggingface_hub import HfApi
    hf_login()
    datasets = HfApi().list_datasets(search=query, limit=limit)
    return [d.id for d in datasets]


def search_hf_models(query: str, limit: int = 20) -> list[str]:
    """Return model IDs from the Hugging Face Hub matching ``query``."""
    from huggingface_hub import HfApi
    hf_login()
    models = HfApi().list_models(search=query, limit=limit)
    return [m.id for m in models]


def lobe_info(marble) -> list[dict]:
    """Return information about all lobes in ``marble``."""
    manager = marble.get_brain().get_lobe_manager()
    manager.update_attention()
    info = []
    for lobe in manager.lobes:
        info.append(
            {
                "id": lobe.id,
                "neurons": len(lobe.neuron_ids),
                "attention": float(lobe.attention_score),
            }
        )
    return info


def add_lobe(marble, neuron_ids: list[int]) -> int:
    """Create a new lobe with ``neuron_ids`` and return its ID."""
    manager = marble.get_brain().get_lobe_manager()
    lobe = manager.genesis([int(n) for n in neuron_ids])
    return lobe.id


def organize_lobes(marble) -> int:
    """Organize neurons into lobes and return the total count."""
    manager = marble.get_brain().get_lobe_manager()
    manager.organize()
    return len(manager.lobes)


def self_attention_lobes(marble, loss: float) -> None:
    """Run self-attention on all lobes using ``loss`` as feedback."""
    manager = marble.get_brain().get_lobe_manager()
    manager.self_attention(float(loss))


def select_high_attention_neurons(marble, threshold: float = 1.0) -> list[int]:
    """Return neuron IDs from lobes with attention above ``threshold``."""
    manager = marble.get_brain().get_lobe_manager()
    ids = manager.select_high_attention(float(threshold))
    return [int(i) for i in ids]


def load_hf_model_wrapper(model_name: str):
    """Return a pretrained model from the Hugging Face Hub."""
    return load_hf_model(model_name)


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
    model = load_hf_model_wrapper(model_name)
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
        if name.startswith("_") or name in {"streamlit_playground", "tests", "setup"}:
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


def list_module_classes(module_name: str) -> list[str]:
    """Return a sorted list of public classes in ``module_name``."""
    module = importlib.import_module(module_name)
    classes = []
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if not name.startswith("_"):
            classes.append(name)
    return sorted(classes)


def search_repository_functions(query: str) -> list[str]:
    """Return function names across the repository matching ``query``."""

    query = query.lower()
    funcs: set[str] = set()
    for name in list_repo_modules() + ["marble_interface"]:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        for fname, obj in inspect.getmembers(module, inspect.isfunction):
            if fname.startswith("_"):
                continue
            if query in fname.lower():
                funcs.add(fname)
    return sorted(funcs)


def find_repository_functions(query: str) -> list[tuple[str, str]]:
    """Return ``(module, function)`` pairs whose names match ``query``.

    The search spans all top-level modules returned by :func:`list_repo_modules`
    as well as ``marble_interface``. Results are sorted alphabetically by module
    then function name.
    """

    q = query.lower()
    matches: list[tuple[str, str]] = []
    for name in list_repo_modules() + ["marble_interface"]:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        for fname, obj in inspect.getmembers(module, inspect.isfunction):
            if fname.startswith("_"):
                continue
            if q in fname.lower():
                matches.append((name, fname))
    return sorted(matches)


def create_module_object(module_name: str, class_name: str, marble=None, **params):
    """Instantiate ``class_name`` from ``module_name`` using ``params``."""
    module = importlib.import_module(module_name)
    if not hasattr(module, class_name):
        raise ValueError(f"Unknown class: {class_name}")
    cls = getattr(module, class_name)
    sig = inspect.signature(cls.__init__)
    kwargs = {}
    for name, p in sig.parameters.items():
        if name == "self":
            continue
        if marble is not None:
            if name == "marble":
                kwargs[name] = marble
                continue
            if name == "core":
                kwargs[name] = marble.get_core()
                continue
            if name in {"nb", "neuronenblitz"}:
                kwargs[name] = marble.get_neuronenblitz()
                continue
            if name == "brain":
                kwargs[name] = marble.get_brain()
                continue
        if name in params:
            kwargs[name] = params[name]
        elif p.default is not inspect.Parameter.empty:
            kwargs[name] = p.default
        else:
            raise ValueError(f"Missing parameter: {name}")
    return cls(**kwargs)


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


def execute_function_sequence(steps: list[dict], marble=None) -> list[object]:
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
            results.append(execute_marble_function(func_name, marble, **params))
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


def move_pipeline_step(
    pipeline: list[dict], old_index: int, new_index: int
) -> list[dict]:
    """Move a pipeline step from ``old_index`` to ``new_index`` and return it."""
    if old_index < 0 or old_index >= len(pipeline):
        raise IndexError("old_index out of range")
    if new_index < 0 or new_index >= len(pipeline):
        raise IndexError("new_index out of range")
    step = pipeline.pop(old_index)
    pipeline.insert(new_index, step)
    return pipeline


def pipeline_to_networkx(pipeline: list[dict]) -> nx.DiGraph:
    """Return a NetworkX graph representing ``pipeline`` sequentially."""
    g = nx.DiGraph()
    for i, step in enumerate(pipeline):
        label = step.get("func")
        module = step.get("module")
        if module:
            label = f"{module}.{label}"
        g.add_node(i, label=f"{i + 1}. {label}", params=step.get("params", {}))
        if i > 0:
            g.add_edge(i - 1, i)
    return g


def pipeline_figure(pipeline: list[dict], layout: str = "spring") -> go.Figure:
    """Return a Plotly figure visualizing the function pipeline."""
    g = pipeline_to_networkx(pipeline)
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
        text.append(g.nodes[n]["label"])
    fig = go.Figure()
    fig.add_scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
    )
    fig.add_scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=text,
        textposition="top center",
        marker=dict(size=10, color="#2ca02c"),
    )
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
    return fig


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


def save_config_yaml(yaml_text: str, path: str) -> None:
    """Write ``yaml_text`` to ``path``.

    Parameters
    ----------
    yaml_text:
        YAML configuration as text.
    path:
        Destination file path.
    """

    with open(path, "w", encoding="utf-8") as f:
        f.write(yaml_text)


def render_config_editor(data: dict, path: str = "", depth: int = 0) -> dict:
    """Render widgets for ``data`` and return updated values.

    To comply with Streamlit's restriction against nested expanders, only the
    top-level keys use :func:`st.expander`. Nested structures are rendered inside
    simple containers instead of additional expanders.
    """

    if isinstance(data, dict):
        out = {}
        for key, val in data.items():
            full = f"{path}.{key}" if path else key
            if depth == 0:
                with st.expander(key, expanded=False):
                    out[key] = render_config_editor(val, full, depth + 1)
            else:
                with st.container():
                    st.markdown(f"**{key}**")
                    out[key] = render_config_editor(val, full, depth + 1)
        return out
    else:
        key = path
        if isinstance(data, bool):
            return st.checkbox(key, value=data, key=key)
        if isinstance(data, int):
            return int(st.number_input(key, value=data, step=1, key=key))
        if isinstance(data, float):
            return float(st.number_input(key, value=data, key=key))
        return st.text_input(key, value=str(data), key=key)


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


def metrics_dataframe(marble) -> pd.DataFrame:
    """Return the metrics collected by ``marble`` as a ``DataFrame``."""
    mv = marble.get_metrics_visualizer()
    metrics = mv.metrics
    max_len = max((len(v) for v in metrics.values()), default=0)
    data = {}
    for k, v in metrics.items():
        pad = [None] * (max_len - len(v))
        data[k] = list(v) + pad
    return pd.DataFrame(data)


def metrics_figure(marble, window_size: int = 10) -> go.Figure:
    """Return a Plotly figure visualizing live metrics from ``marble``."""
    mv = marble.get_metrics_visualizer()
    fig = go.Figure()
    for name, values in mv.metrics.items():
        if not values:
            continue
        smooth = []
        for i in range(len(values)):
            start = max(0, i - window_size + 1)
            smooth.append(sum(values[start : i + 1]) / (i - start + 1))
        fig.add_scatter(y=smooth, mode="lines", name=name)
    fig.update_layout(xaxis_title="Updates", yaxis_title="Value")
    return fig


def system_stats(device: int = 0) -> dict:
    """Return current CPU and GPU memory usage in megabytes."""
    from system_metrics import get_gpu_memory_usage, get_system_memory_usage

    return {
        "ram_mb": get_system_memory_usage(),
        "gpu_mb": get_gpu_memory_usage(device),
    }


def core_statistics(marble) -> dict:
    """Return neuron, synapse and tier counts for ``marble``."""
    core = marble.get_core()
    tiers = {n.tier for n in core.neurons}
    return {
        "neurons": len(core.neurons),
        "synapses": len(core.synapses),
        "tiers": len(tiers),
    }


def core_weight_matrix(core, limit: int | None = None) -> np.ndarray:
    """Return a synaptic weight matrix for ``core``."""

    n = len(core.neurons)
    mat = np.zeros((n, n), dtype=np.float32)
    for s in core.synapses:
        if s.source < n and s.target < n:
            mat[s.source, s.target] = float(s.weight)
    if limit is not None:
        mat = mat[:limit, :limit]
    return mat


def core_heatmap_figure(
    core, limit: int | None = None, color_scale: str = "Viridis"
) -> go.Figure:
    """Return a Plotly heatmap visualizing synaptic weights."""

    mat = core_weight_matrix(core, limit=limit)
    fig = go.Figure(data=go.Heatmap(z=mat, colorscale=color_scale))
    fig.update_layout(
        xaxis_title="Target Neuron",
        yaxis_title="Source Neuron",
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig


def get_neuromod_state(marble) -> dict:
    """Return the current neuromodulatory signal levels."""
    return marble.brain.neuromodulatory_system.get_context()


def set_neuromod_state(
    marble,
    arousal: float | None = None,
    stress: float | None = None,
    reward: float | None = None,
    emotion: str | None = None,
) -> dict:
    """Update neuromodulatory signals and return the new state."""
    updates: dict[str, object] = {}
    if arousal is not None:
        updates["arousal"] = float(arousal)
    if stress is not None:
        updates["stress"] = float(stress)
    if reward is not None:
        updates["reward"] = float(reward)
    if emotion is not None:
        updates["emotion"] = str(emotion)
    if updates:
        marble.brain.neuromodulatory_system.update_signals(**updates)
    return marble.brain.neuromodulatory_system.get_context()


def load_readme() -> str:
    """Return the repository ``README.md`` contents."""
    path = os.path.join(os.path.dirname(__file__), "README.md")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_tutorial() -> str:
    """Return the ``TUTORIAL.md`` contents."""
    path = os.path.join(os.path.dirname(__file__), "TUTORIAL.md")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def list_documentation_files() -> list[str]:
    """Return available markdown documentation files."""
    root = os.path.dirname(__file__)
    files = [f for f in os.listdir(root) if f.endswith(".md")]
    return sorted(files)


def load_documentation(doc_name: str) -> str:
    """Return the contents of ``doc_name`` located in the repository root."""
    path = os.path.join(os.path.dirname(__file__), doc_name)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_module_source(module_name: str) -> str:
    """Return source code for ``module_name``."""
    module = importlib.import_module(module_name)
    return inspect.getsource(module)


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


if "run_example_project" not in globals():

    def run_example_project(project_name: str) -> str:
        """Execute an example project script and return captured output."""
        ex_dir = os.path.join(os.path.dirname(__file__), "examples")
        path = os.path.join(ex_dir, project_name)
        out_buf = io.StringIO()
        err_buf = io.StringIO()
        with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
            runpy.run_path(path, run_name="__main__")
        return out_buf.getvalue() + err_buf.getvalue()


def start_remote_server(
    host: str = "localhost",
    port: int = 8000,
    remote_url: str | None = None,
    compression_level: int = 6,
    compression_enabled: bool = True,
    auth_token: str | None = None,
) -> object:
    """Start and return a ``RemoteBrainServer`` instance."""
    from remote_offload import RemoteBrainServer

    server = RemoteBrainServer(
        host=host,
        port=port,
        remote_url=remote_url,
        compression_level=compression_level,
        compression_enabled=compression_enabled,
        auth_token=auth_token,
    )
    server.start()
    return server


def create_remote_client(
    url: str,
    timeout: float = 5.0,
    max_retries: int = 3,
    compression_level: int = 6,
    compression_enabled: bool = True,
    auth_token: str | None = None,
) -> object:
    """Return a configured ``RemoteBrainClient``."""
    from remote_offload import RemoteBrainClient

    return RemoteBrainClient(
        url,
        timeout=timeout,
        max_retries=max_retries,
        compression_level=compression_level,
        compression_enabled=compression_enabled,
        auth_token=auth_token,
    )


def create_torrent_system(
    client_id: str = "main",
    buffer_size: int = 10,
    heartbeat_interval: int = 30,
) -> tuple[object, object]:
    """Return a tracker and connected ``BrainTorrentClient``."""
    from torrent_offload import BrainTorrentClient, BrainTorrentTracker

    tracker = BrainTorrentTracker()
    client = BrainTorrentClient(
        client_id,
        tracker,
        buffer_size=buffer_size,
        heartbeat_interval=heartbeat_interval,
    )
    client.connect()
    return tracker, client


def list_learner_modules() -> list[str]:
    """Return module names that define ``*Learner`` or ``*Agent`` classes."""
    root = os.path.dirname(__file__)
    modules = set()
    for mod in pkgutil.iter_modules([root]):
        if (
            mod.ispkg
            or mod.name.startswith("_")
            or mod.name in {"streamlit_playground", "tests", "examples", "setup"}
        ):
            continue
        try:
            module = importlib.import_module(mod.name)
        except ImportError:
            continue
        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__name__.endswith("Learner") or obj.__name__.endswith("Agent"):
                modules.add(mod.name)
                break
    return sorted(modules)


def list_learner_classes(module_name: str) -> list[str]:
    """Return learner class names in ``module_name``."""
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return []
    classes = []
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if name.endswith("Learner") or name.endswith("Agent"):
            classes.append(name)
    return sorted(classes)


def create_learner(module_name: str, class_name: str, marble, **params):
    """Instantiate a learner class using components from ``marble``."""
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(f"Could not import {module_name}") from e
    if not hasattr(module, class_name):
        raise ValueError(f"Unknown class: {class_name}")
    cls = getattr(module, class_name)
    sig = inspect.signature(cls.__init__)
    kwargs = {}
    for name, p in sig.parameters.items():
        if name == "self":
            continue
        if name == "core":
            kwargs[name] = marble.get_core()
            continue
        if name in {"nb", "neuronenblitz"}:
            kwargs[name] = marble.get_neuronenblitz()
            continue
        if name in params:
            kwargs[name] = params[name]
        elif p.default is not inspect.Parameter.empty:
            kwargs[name] = p.default
        else:
            raise ValueError(f"Missing parameter: {name}")
    return cls(**kwargs)


def train_learner(learner, samples, epochs: int = 1) -> None:
    """Train ``learner`` on ``samples`` for ``epochs``."""
    if hasattr(learner, "train"):
        sig = inspect.signature(learner.train)
        if "epochs" in sig.parameters:
            learner.train(samples, epochs=epochs)
        else:
            for _ in range(int(epochs)):
                learner.train(samples)
        return
    if hasattr(learner, "train_step"):
        for _ in range(int(epochs)):
            for sample in samples:
                if isinstance(sample, tuple):
                    learner.train_step(*sample)
                else:
                    learner.train_step(sample)
        return
    raise ValueError("Learner has no train or train_step method")


def start_background_training(
    marble,
    samples: list[tuple] | list,
    epochs: int = 1,
    validation_samples: list[tuple] | list | None = None,
) -> None:
    """Start training ``marble`` in a background thread."""

    marble.get_brain().start_training(
        samples, epochs=epochs, validation_examples=validation_samples
    )


def wait_for_training(marble) -> None:
    """Block until ``marble`` finishes background training."""

    marble.get_brain().wait_for_training()


def training_in_progress(marble) -> bool:
    """Return ``True`` if ``marble`` is currently training."""

    return bool(marble.get_brain().training_active)


def create_gridworld_env(size: int = 4):
    """Return a GridWorld environment of ``size`` x ``size``."""

    from reinforcement_learning import GridWorld

    return GridWorld(size=int(size))


def run_gridworld_episode(
    marble,
    episodes: int = 10,
    max_steps: int = 50,
    size: int = 4,
    double_q: bool = False,
) -> list[float]:
    """Train a :class:`MarbleQLearningAgent` in GridWorld and return rewards."""

    from reinforcement_learning import MarbleQLearningAgent, train_gridworld

    env = create_gridworld_env(size)
    agent = MarbleQLearningAgent(
        marble.get_core(), marble.get_neuronenblitz(), double_q=bool(double_q)
    )
    rewards = train_gridworld(
        agent,
        env,
        episodes=int(episodes),
        max_steps=int(max_steps),
    )
    return [float(r) for r in rewards]


def wander_neuronenblitz(
    marble,
    input_value: float | str | bytes,
    apply_plasticity: bool = True,
) -> tuple[float, list[int]]:
    """Run ``Neuronenblitz.dynamic_wander`` and return output and path.

    Parameters
    ----------
    marble:
        Active MARBLE instance.
    input_value:
        Value provided to the wanderer. Can be numeric or any value accepted by
        :func:`_parse_value` when used through the playground.
    apply_plasticity:
        Whether synaptic plasticity should be applied while wandering.

    Returns
    -------
    tuple
        Output value and list of neuron IDs visited in order.
    """

    nb = marble.get_neuronenblitz()
    out, path = nb.dynamic_wander(input_value, apply_plasticity=bool(apply_plasticity))
    ids = [int(s.target) for s in path]
    return float(out), ids


def parallel_wander_neuronenblitz(
    marble,
    input_value: float | str | bytes,
    processes: int = 2,
) -> list[tuple[float, int]]:
    """Run ``dynamic_wander_parallel`` and return ``(output, seed)`` pairs."""

    nb = marble.get_neuronenblitz()
    results = nb.dynamic_wander_parallel(input_value, num_processes=int(processes))
    return [(float(o), int(s)) for o, s in results]


def create_hybrid_memory(
    marble,
    vector_path: str = "vector_store.pkl",
    symbolic_path: str = "symbolic_memory.pkl",
) -> object:
    """Create ``HybridMemory`` for ``marble`` and return it."""

    from hybrid_memory import HybridMemory

    hm = HybridMemory(
        marble.get_core(),
        marble.get_neuronenblitz(),
        vector_path,
        symbolic_path,
    )
    marble.hybrid_memory = hm
    return hm


def hybrid_memory_store(marble, key: str, value: float | str | bytes) -> None:
    """Store ``value`` in MARBLE's hybrid memory under ``key``."""

    if marble.hybrid_memory is None:
        raise ValueError("Hybrid memory not initialized")
    marble.hybrid_memory.store(key, _parse_value(value))


def hybrid_memory_retrieve(
    marble, query: float | str | bytes, top_k: int = 3
) -> list[tuple[object, object]]:
    """Return top ``k`` items matching ``query`` from hybrid memory."""

    if marble.hybrid_memory is None:
        raise ValueError("Hybrid memory not initialized")
    return marble.hybrid_memory.retrieve(_parse_value(query), top_k=int(top_k))


def hybrid_memory_forget(marble, max_entries: int = 1000) -> None:
    """Remove old entries keeping at most ``max_entries``."""

    if marble.hybrid_memory is None:
        raise ValueError("Hybrid memory not initialized")
    marble.hybrid_memory.forget_old(max_entries=int(max_entries))


def meta_controller_info(marble) -> dict:
    """Return parameters and loss history of the meta-controller."""

    mc = marble.get_brain().meta_controller
    return {
        "history_length": int(mc.history_length),
        "adjustment": float(mc.adjustment),
        "min_threshold": float(mc.min_threshold),
        "max_threshold": float(mc.max_threshold),
        "loss_history": list(mc.loss_history),
        "plasticity_threshold": float(marble.get_neuronenblitz().plasticity_threshold),
    }


def update_meta_controller(
    marble,
    history_length: int | None = None,
    adjustment: float | None = None,
    min_threshold: float | None = None,
    max_threshold: float | None = None,
) -> dict:
    """Update meta-controller parameters and return the new settings."""

    mc = marble.get_brain().meta_controller
    if history_length is not None:
        mc.history_length = int(history_length)
        mc.loss_history = mc.loss_history[-mc.history_length :]
    if adjustment is not None:
        mc.adjustment = float(adjustment)
    if min_threshold is not None:
        mc.min_threshold = float(min_threshold)
    if max_threshold is not None:
        mc.max_threshold = float(max_threshold)
    return meta_controller_info(marble)


def adjust_meta_controller(marble) -> float:
    """Apply the meta-controller adjustment and return the new threshold."""

    mc = marble.get_brain().meta_controller
    mc.adjust(marble.get_neuronenblitz())
    return float(marble.get_neuronenblitz().plasticity_threshold)


def reset_meta_loss_history(marble) -> None:
    """Clear the meta-controller loss history."""

    marble.get_brain().meta_controller.loss_history.clear()


def super_evo_history(marble) -> list[dict]:
    """Return recorded metrics from the super evolution controller."""

    se = marble.get_brain().super_evo_controller
    return [] if se is None else list(se.history)


def super_evo_changes(marble) -> list[dict]:
    """Return parameter change log from the super evolution controller."""

    se = marble.get_brain().super_evo_controller
    return [] if se is None else list(se.change_log)


def clear_super_evo_changes(marble) -> None:
    """Clear the super evolution change log."""

    se = marble.get_brain().super_evo_controller
    if se is not None:
        se.change_log.clear()


def run_dimensional_search(marble, loss: float) -> int:
    """Evaluate ``DimensionalitySearch`` using ``loss`` and return rep size."""

    ds = marble.get_brain().dim_search
    if ds is None:
        raise ValueError("DimensionalitySearch not enabled")
    ds.evaluate(float(loss))
    return int(marble.get_core().rep_size)


def run_nd_topology(marble, loss: float) -> int:
    """Evaluate ``NDimensionalTopologyManager`` and return rep size."""

    ndt = marble.get_brain().nd_topology
    if ndt is None:
        raise ValueError("NDimensionalTopologyManager not enabled")
    ndt.evaluate(float(loss))
    return int(marble.get_core().rep_size)


def start_auto_firing(marble, interval_ms: int = 1000) -> None:
    """Start MARBLE's auto-firing thread."""

    marble.get_brain().firing_interval_ms = int(interval_ms)
    marble.get_brain().start_auto_firing()


def stop_auto_firing(marble) -> None:
    """Stop MARBLE's auto-firing thread."""

    marble.get_brain().stop_auto_firing()


def list_test_files(pattern: str = "tests/test_*.py") -> list[str]:
    """Return available pytest files matching ``pattern``."""
    root = os.path.dirname(__file__)
    paths = glob.glob(os.path.join(root, pattern))
    return sorted(os.path.basename(p) for p in paths)


def run_tests(pattern: str | None = None) -> str:
    """Run ``pytest`` with an optional ``-k`` pattern and return output."""
    args: list[str] = []
    if pattern:
        args.extend(["-k", pattern])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        code = pytest.main(args)
    return buf.getvalue() + f"\nExit code: {code}\n"


def _persist_ui_state() -> None:
    """Persist tabs, expanders and scroll position across reruns using JS."""

    script = """
    <script>
    const doc = window.parent.document;

    // scroll position
    const pos = sessionStorage.getItem('scrollPos') || 0;
    window.scrollTo(0, parseFloat(pos));
    window.addEventListener('scroll', () => {
        sessionStorage.setItem('scrollPos', doc.documentElement.scrollTop || window.pageYOffset);
    });

    // active tab index
    const tabs = doc.querySelectorAll('[data-baseweb="tab"]');
    const savedTab = sessionStorage.getItem('activeTabIndex');
    if (savedTab !== null && tabs[savedTab]) {
        tabs[savedTab].click();
    }
    tabs.forEach((t, idx) => {
        t.addEventListener('click', () => {
            sessionStorage.setItem('activeTabIndex', idx);
        });
    });

    // expander state
    const expanders = doc.querySelectorAll('[data-testid="stExpander"] details');
    expanders.forEach((exp, idx) => {
        const open = sessionStorage.getItem('expander' + idx);
        if (open === 'true') exp.open = true;
        exp.addEventListener('toggle', () => {
            sessionStorage.setItem('expander' + idx, exp.open);
        });
    });
    </script>
    """

    components.html(script, height=0, width=0)


def _detect_client_settings() -> None:
    """Detect client theme and screen size via JavaScript."""

    js = """
    <script>
    const theme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    const mobile = window.innerWidth < 768 ? '1' : '0';
    const url = new URL(window.location);
    let changed = false;
    if (url.searchParams.get('theme') !== theme) { url.searchParams.set('theme', theme); changed = true; }
    if (url.searchParams.get('mobile') !== mobile) { url.searchParams.set('mobile', mobile); changed = true; }
    if (changed) { window.location.replace(url); }
    </script>
    """
    components.html(js, height=0)


def run_playground() -> None:
    """Launch the Streamlit MARBLE playground."""
    st.set_page_config(page_title="MARBLE Playground", layout="wide")
    _detect_client_settings()
    _detect_device()
    params = st.experimental_get_query_params()
    theme = params.get("theme", ["light"])[0]
    mobile = params.get("mobile", ["0"])[0] == "1"
    st.session_state["theme"] = theme
    st.session_state["mobile"] = mobile
    st.session_state.setdefault("device", "mobile" if mobile else "desktop")
    if theme == "dark":
        st.markdown(
            "<style>body{background-color:#0e1117;color:#d0d0d0;}</style>",
            unsafe_allow_html=True,
        )
    if mobile:
        st.markdown(
            "<style>div.block-container{padding:0.5rem;}"
            "div[data-testid='stSidebar']{width:100%;}</style>",
            unsafe_allow_html=True,
        )
    st.title("MARBLE Playground")

    if st.button("About"):
        if hasattr(st, "dialog"):
            with st.dialog("About MARBLE"):
                st.markdown(load_readme())
        else:
            with st.expander("About MARBLE", expanded=True):
                st.markdown(load_readme())

    if "registry" not in st.session_state:
        st.session_state["registry"] = MarbleRegistry()
    registry: MarbleRegistry = st.session_state["registry"]
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
    inst_name = st.sidebar.text_input("Instance Name", value=registry.active or "main")
    if st.sidebar.button("Create Instance"):
        yaml_data = None
        if cfg_upload is not None:
            yaml_data = cfg_upload.getvalue().decode("utf-8")
        elif cfg_text.strip():
            yaml_data = cfg_text
        else:
            with open(cfg_path, "r", encoding="utf-8") as f:
                yaml_data = f.read()
        marble = registry.create(
            inst_name, cfg_path if not yaml_data else None, yaml_text=yaml_data
        )
        st.session_state["marble"] = marble
        st.session_state["config_yaml"] = yaml_data
        st.sidebar.success(f"Instance '{inst_name}' created")

    if registry.list():
        active = st.sidebar.selectbox(
            "Active Instance",
            registry.list(),
            index=registry.list().index(registry.active or registry.list()[0]),
            key="active_instance",
        )
        if st.sidebar.button("Switch Instance"):
            registry.set_active(active)
            st.session_state["marble"] = registry.get()
        if st.sidebar.button("Duplicate Instance"):
            base = active
            new_name = base + "_copy"
            counter = 1
            while new_name in registry.list():
                new_name = f"{base}_copy{counter}"
                counter += 1
            registry.duplicate(base, new_name)
            st.sidebar.success(f"Instance duplicated as {new_name}")
        if st.sidebar.button("Delete Instance"):
            registry.delete(active)
            if registry.active:
                st.session_state["marble"] = registry.get()
            else:
                st.session_state["marble"] = None
            st.sidebar.success("Instance deleted")

    save_path = st.sidebar.text_input("Save Path", "marble.pkl")
    if st.sidebar.button("Save MARBLE") and st.session_state.get("marble") is not None:
        save_marble_system(st.session_state["marble"], save_path)
        registry.instances[registry.active] = st.session_state["marble"]
        st.sidebar.success("Model saved")

    load_file = st.sidebar.file_uploader("Load MARBLE", type=["pkl"], key="load_marble")
    if st.sidebar.button("Load MARBLE") and load_file is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(load_file.read())
        tmp.close()
        st.session_state["marble"] = load_marble_system(tmp.name)
        registry.instances[registry.active] = st.session_state["marble"]
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
        registry.instances[registry.active] = st.session_state["marble"]
        st.sidebar.success("Core loaded")

    if "config_yaml" in st.session_state:
        with st.sidebar.expander("Current Config"):
            st.code(st.session_state["config_yaml"], language="yaml")
            st.download_button(
                "Download config.yaml",
                st.session_state["config_yaml"],
                file_name="config.yaml",
            )
            cfg_save_path = st.text_input(
                "Save Config Path", "config_saved.yaml", key="cfg_save_path"
            )
            if st.button("Save Config File", key="cfg_save_btn"):
                save_config_yaml(st.session_state["config_yaml"], cfg_save_path)
                st.sidebar.success("Config saved")
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
    if "hf_search_results" not in st.session_state:
        st.session_state["hf_search_results"] = []
    hf_query = st.sidebar.text_input("HF Dataset Search", key="hf_query")
    if st.sidebar.button("Search Datasets", key="hf_do_search") and hf_query:
        st.session_state["hf_search_results"] = search_hf_datasets(hf_query)
        st.sidebar.success(
            f"Found {len(st.session_state['hf_search_results'])} datasets"
        )
    if st.session_state["hf_search_results"]:
        choice = st.sidebar.selectbox(
            "Search Results", st.session_state["hf_search_results"], key="hf_choice"
        )
        if st.sidebar.button("Select Dataset", key="hf_select"):
            st.session_state["hf_dataset_name"] = choice
            st.sidebar.success("Dataset selected")
    hf_name = st.sidebar.text_input(
        "HF Dataset Name", value=st.session_state.get("hf_dataset_name", "")
    )
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
            progress_bar = st.sidebar.progress(0.0)

            def _cb(p):
                progress_bar.progress(min(p, 1.0))

            train_marble_system(
                marble,
                examples,
                epochs=epochs,
                progress_callback=_cb,
            )
            progress_bar.empty()
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
        (
            tab_search,
            tab_iface,
            tab_mod,
            tab_cls,
            tab_learner,
            tab_autoenc,
            tab_pipe,
            tab_code,
            tab_vis,
            tab_heat,
            tab_metrics,
            tab_stats,
            tab_neuro,
            tab_lobes,
            tab_core,
            tab_cfg,
            tab_model,
            tab_offload,
            tab_async,
            tab_rl,
            tab_adapt,
            tab_memory,
            tab_nbexp,
            tab_proj,
            tab_tests,
            tab_docs,
            tab_browser,
            tab_src,
        ) = st.tabs(
            [
                "Function Search",
                "marble_interface",
                "Modules",
                "Classes",
                "Learners",
                "Autoencoder",
                "Pipeline",
                "Custom Code",
                "Visualization",
                "Weight Heatmap",
                "Metrics",
                "System Stats",
                "Neuromodulation",
                "Lobe Manager",
                "Core Tools",
                "Config Editor",
                "Model Conversion",
                "Offloading",
                "Async Training",
                "RL Sandbox",
                "Adaptive Control",
                "Hybrid Memory",
                "NB Explorer",
                "Projects",
                "Tests",
                "Documentation",
                "Dataset Browser",
                "Source Browser",
            ]
        )

        with tab_search:
            query = st.text_input("Search", key="repo_search_query")
            results = find_repository_functions(query) if query else []
            if results:
                options = [f"{m}.{f}" for m, f in results]
                choice = st.selectbox("Function", options, key="repo_search_func")
                module, func_name = choice.split(".", 1)
                func_obj = getattr(importlib.import_module(module), func_name)
                doc = inspect.getdoc(func_obj) or ""
                if doc:
                    st.markdown(doc)
                sig = inspect.signature(func_obj)
                inputs = {}
                for name, param in sig.parameters.items():
                    if name == "marble":
                        continue
                    default = (
                        None
                        if param.default is inspect.Parameter.empty
                        else param.default
                    )
                    ann = param.annotation
                    widget = None
                    if ann is bool or isinstance(default, bool):
                        widget = st.checkbox(
                            name,
                            value=bool(default) if default is not None else False,
                            key=f"repo_{name}",
                        )
                    elif ann in (int, float) or isinstance(default, (int, float)):
                        val = 0 if default is None else float(default)
                        widget = st.number_input(name, value=val, key=f"repo_{name}")
                    else:
                        widget = st.text_input(
                            name,
                            value="" if default is None else str(default),
                            key=f"repo_{name}",
                        )
                    inputs[name] = widget
                if st.button("Execute", key="repo_execute"):
                    parsed = {}
                    for k, v in inputs.items():
                        if isinstance(v, str) and v != "":
                            try:
                                parsed[k] = json.loads(v)
                            except Exception:
                                parsed[k] = _parse_value(v)
                        else:
                            parsed[k] = v
                    if module == "marble_interface":
                        result = execute_marble_function(func_name, marble, **parsed)
                    else:
                        result = execute_module_function(
                            module, func_name, marble, **parsed
                        )
                    st.write(result)

        with tab_iface:
            funcs = list_marble_functions()
            iface_filter = st.text_input("Search", key="iface_search")
            if iface_filter:
                funcs = [f for f in funcs if iface_filter.lower() in f.lower()]
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
            mod_filter = st.text_input("Search", key="mod_search")
            if mod_filter:
                funcs = [f for f in funcs if mod_filter.lower() in f.lower()]
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

        with tab_cls:
            st.write("Instantiate any class from repository modules.")
            cls_module = st.selectbox("Module", list_repo_modules(), key="cls_mod")
            classes = list_module_classes(cls_module)
            cls_choice = st.selectbox("Class", classes, key="cls_choice")
            cls_obj = getattr(importlib.import_module(cls_module), cls_choice)
            sig = inspect.signature(cls_obj.__init__)
            cparams = {}
            for name, param in sig.parameters.items():
                if name == "self":
                    continue
                default = (
                    None if param.default is inspect.Parameter.empty else param.default
                )
                cparams[name] = st.text_input(
                    name,
                    value="" if default is None else str(default),
                    key=f"cls_{name}",
                )
            if st.button("Create Instance", key="cls_create"):
                parsed = {}
                for k, v in cparams.items():
                    if v == "":
                        continue
                    try:
                        parsed[k] = json.loads(v)
                    except Exception:
                        try:
                            parsed[k] = float(v)
                        except Exception:
                            parsed[k] = v
                obj = create_module_object(cls_module, cls_choice, marble, **parsed)
                st.session_state["last_object"] = obj
                st.success(f"Created {cls_choice}")
                st.write(repr(obj))

        with tab_learner:
            st.write("Create and train built-in learner classes.")
            if "active_learner" not in st.session_state:
                st.session_state["active_learner"] = None
            learn_mods = list_learner_modules()
            mod_sel = st.selectbox("Module", learn_mods, key="learn_mod")
            classes = list_learner_classes(mod_sel)
            cls_sel = st.selectbox("Learner Class", classes, key="learn_cls")
            cls_obj = getattr(importlib.import_module(mod_sel), cls_sel)
            sig = inspect.signature(cls_obj)
            lparams = {}
            for name, param in sig.parameters.items():
                if name in {"self", "core", "nb", "neuronenblitz"}:
                    continue
                default = (
                    None if param.default is inspect.Parameter.empty else param.default
                )
                widget = st.text_input(
                    name,
                    value="" if default is None else str(default),
                    key=f"learn_{name}",
                )
                lparams[name] = widget
            if st.button("Create Learner", key="create_learner"):
                parsed = {}
                for k, v in lparams.items():
                    if v == "":
                        continue
                    try:
                        parsed[k] = json.loads(v)
                    except Exception:
                        try:
                            parsed[k] = float(v)
                        except Exception:
                            parsed[k] = v
                st.session_state["active_learner"] = create_learner(
                    mod_sel, cls_sel, marble, **parsed
                )
                st.success("Learner created")
            if st.session_state.get("active_learner") is not None:
                lfile = st.file_uploader(
                    "Training Data",
                    type=["csv", "json", "jsonl", "zip"],
                    key="learn_data",
                )
                lepochs = st.number_input(
                    "Epochs", value=1, min_value=1, step=1, key="learn_epochs"
                )
                if st.button("Train", key="train_learner") and lfile is not None:
                    examples = load_examples(lfile)
                    train_learner(
                        st.session_state["active_learner"],
                        examples,
                        epochs=int(lepochs),
                    )
                    st.success("Training complete")

        with tab_autoenc:
            st.write("Train a denoising autoencoder on numeric values.")
            file = st.file_uploader(
                "Values Dataset",
                type=["csv", "json", "jsonl", "zip", "txt"],
                key="auto_vals",
            )
            a_epochs = st.number_input(
                "Epochs", value=1, min_value=1, step=1, key="auto_epochs"
            )
            a_std = st.number_input(
                "Noise Std", value=0.1, format="%.2f", key="auto_std"
            )
            a_decay = st.number_input(
                "Noise Decay", value=0.99, format="%.2f", step=0.01, key="auto_decay"
            )
            if st.button("Train Autoencoder", key="auto_train") and file is not None:
                values = load_value_list(file)
                loss = train_autoencoder(
                    marble,
                    values,
                    epochs=int(a_epochs),
                    noise_std=float(a_std),
                    noise_decay=float(a_decay),
                )
                st.success(f"Training complete. Final loss: {loss:.6f}")

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
                    if (
                        cols[2].button("⬇", key=f"pipe_down_{i}")
                        and i < len(st.session_state["pipeline"]) - 1
                    ):
                        move_pipeline_step(st.session_state["pipeline"], i, i + 1)
                    if cols[3].button("✕", key=f"pipe_rm_{i}"):
                        remove_pipeline_step(st.session_state["pipeline"], i)
                        st.rerun()
            with st.expander("Load/Save Pipeline"):
                pipe_up = st.file_uploader(
                    "Load Pipeline", type=["json"], key="pipe_load"
                )
                if st.button("Load", key="load_pipe") and pipe_up is not None:
                    st.session_state["pipeline"] = load_pipeline_from_json(pipe_up)
                    st.success("Pipeline loaded")
                if st.session_state["pipeline"]:
                    pipe_json = json.dumps(st.session_state["pipeline"], indent=2)
                    st.download_button(
                        "Download pipeline.json", pipe_json, file_name="pipeline.json"
                    )
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
                        None
                        if param.default is inspect.Parameter.empty
                        else param.default
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
                        widget = st.number_input(name, value=val, key=f"pipe_{name}")
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
            if st.button("Show Pipeline Graph") and st.session_state["pipeline"]:
                fig = pipeline_figure(st.session_state["pipeline"])
                st.plotly_chart(fig, use_container_width=True)
            with st.expander("Step Visualisation"):
                for i, step in enumerate(st.session_state["pipeline"]):
                    module = step.get("module") or "marble_interface"
                    st.markdown(
                        f"**Step {i+1}:** `{module}.{step['func']}`"
                    )
                    params = step.get("params", {})
                    if params:
                        st.write("Parameters:")
                        st.json(params)
                        for name, value in params.items():
                            if isinstance(value, BitTensorDataset):
                                st.write(f"Dataset `{name}` summary:")
                                st.json(value.summary())
                    else:
                        st.write("Parameters: none")
            if st.button("Run Pipeline") and st.session_state["pipeline"]:
                is_mobile = st.session_state.get("device", "desktop") == "mobile" or st.session_state.get("mobile")
                placeholder = st.empty()
                bar = placeholder.progress(0.0) if not is_mobile else None
                text_box = placeholder if is_mobile else st.empty()

                def _on_progress(name, data):
                    pct = (data["index"] + (1 if data["status"] == "completed" else 0)) / data["total"]
                    msg = f"{data['status']}: {data['step']} ({data['device']})"
                    st.session_state["last_progress"] = msg
                    if bar:
                        bar.progress(pct, text=msg)
                    else:
                        text_box.markdown(f"{int(pct*100)}% {msg}")

                global_event_bus.subscribe(_on_progress, events=[PROGRESS_EVENT])
                pipeline_obj = Pipeline(st.session_state["pipeline"])
                res = pipeline_obj.execute(marble)
                for out in res:
                    st.write(out)
                if bar:
                    bar.progress(1.0, text="Pipeline complete")
                else:
                    text_box.markdown("Pipeline complete")
            if st.button("Clear Pipeline"):
                st.session_state["pipeline"] = []

        with tab_code:
            st.write(
                "Execute custom Python code. Use the `marble` variable to access the system."
            )
            code = st.text_area("Code", height=200, key="custom_code")
            if st.button("Run Code", key="run_custom_code") and code.strip():
                out = run_custom_code(code, marble)
                st.write(out)

        with tab_vis:
            st.write("Visualize the current MARBLE core.")
            auto = st.checkbox("Auto Refresh Graph", key="auto_graph")
            interval = st.number_input(
                "Refresh interval (s)",
                min_value=1,
                value=5,
                step=1,
                key="graph_interval",
            )
            layout = st.selectbox("Layout", ["spring", "circular"], key="graph_layout")
            container = st.empty()
            if auto:
                _auto_refresh(interval * 1000, "graph_refresh")
                fig = core_figure(marble.get_core(), layout=layout)
                container.plotly_chart(fig, use_container_width=True)
            elif st.button("Generate Graph", key="show_graph"):
                fig = core_figure(marble.get_core(), layout=layout)
                container.plotly_chart(fig, use_container_width=True)
            if st.button("Show Activations", key="show_acts"):
                activations = {n.id: float(n.value) for n in marble.get_core().neurons}
                fig = activation_figure(marble.get_core(), activations, layout=layout)
                st.plotly_chart(fig, use_container_width=True)

        with tab_heat:
            st.write("Display a heatmap of synaptic weights.")
            limit = st.number_input(
                "Max Neurons", min_value=1, value=100, step=1, key="heat_limit"
            )
            color = st.selectbox(
                "Color Scale", ["Viridis", "Cividis", "Plasma"], key="heat_color"
            )
            if st.button("Generate Heatmap", key="show_heatmap"):
                fig = core_heatmap_figure(
                    marble.get_core(), limit=int(limit), color_scale=color
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab_metrics:
            st.write("Live metrics from the current run.")
            fig = metrics_figure(marble)
            st.plotly_chart(fig, use_container_width=True)
            st.button("Refresh", key="metrics_refresh")
            with st.expander("Event Log"):
                events = marble.get_metrics_visualizer().events
                for name, data in events[-100:]:
                    st.write(f"{name}: {data}")

        with tab_stats:
            st.write("System resource usage in megabytes.")
            stats = system_stats()
            st.metric("RAM", f"{stats['ram_mb']:.1f} MB")
            st.metric("GPU", f"{stats['gpu_mb']:.1f} MB")
            if st.button("Refresh Stats", key="stats_refresh"):
                st.rerun()

        with tab_neuro:
            st.write("Adjust neuromodulatory signals.")
            state = get_neuromod_state(marble)
            arousal = st.slider(
                "Arousal",
                min_value=0.0,
                max_value=1.0,
                value=float(state.get("arousal", 0.0)),
                step=0.01,
            )
            stress = st.slider(
                "Stress",
                min_value=0.0,
                max_value=1.0,
                value=float(state.get("stress", 0.0)),
                step=0.01,
            )
            reward = st.slider(
                "Reward",
                min_value=0.0,
                max_value=1.0,
                value=float(state.get("reward", 0.0)),
                step=0.01,
            )
            emotion = st.text_input(
                "Emotion", value=str(state.get("emotion", "neutral"))
            )
            if st.button("Update Signals", key="neuro_update"):
                new_state = set_neuromod_state(
                    marble,
                    arousal=arousal,
                    stress=stress,
                    reward=reward,
                    emotion=emotion,
                )
                st.success(
                    f"Signals updated: arousal={new_state['arousal']}, stress={new_state['stress']}, reward={new_state['reward']}, emotion={new_state['emotion']}"
                )

        with tab_lobes:
            st.write("Inspect and control lobes.")
            st.dataframe(pd.DataFrame(lobe_info(marble)), use_container_width=True)
            with st.expander("Create Lobe"):
                ids_str = st.text_input("Neuron IDs (comma)", key="lobe_ids")
                if st.button("Create Lobe", key="lobe_create"):
                    ids = [int(x.strip()) for x in ids_str.split(",") if x.strip()]
                    lid = add_lobe(marble, ids)
                    st.success(f"Lobe {lid} created")
            if st.button("Organize Lobes", key="lobe_org"):
                count = organize_lobes(marble)
                st.success(f"{count} lobes organized")
            loss_val = st.number_input(
                "Self-Attention Loss", value=0.0, format="%f", key="lobe_loss"
            )
            if st.button("Apply Self-Attention", key="lobe_self"):
                self_attention_lobes(marble, float(loss_val))
                st.success("Self-attention applied")
            thresh = st.number_input(
                "Select Threshold", value=1.0, format="%f", key="lobe_thresh"
            )
            if st.button("Select High Attention", key="lobe_select"):
                ids = select_high_attention_neurons(marble, float(thresh))
                st.write(ids)

        with tab_core:
            st.write("Manipulate and inspect the MARBLE core.")
            stats = core_statistics(marble)
            st.metric("Neurons", stats["neurons"])
            st.metric("Synapses", stats["synapses"])
            st.metric("Tiers", stats["tiers"])

            st.subheader("Expand Core")
            n_neurons = st.number_input(
                "New Neurons", value=10, step=1, key="exp_neurons"
            )
            n_synapses = st.number_input(
                "New Synapses", value=15, step=1, key="exp_synapses"
            )
            alt_prob = st.number_input(
                "Alt Connection Prob", value=0.1, step=0.1, format="%f", key="exp_alt"
            )
            tier = st.text_input("Target Tier", "", key="exp_tier")
            ntypes = st.text_input("Neuron Types (comma)", "", key="exp_types")
            if st.button("Expand", key="do_expand"):
                types = [s.strip() for s in ntypes.split(",") if s.strip()] or None
                expand_marble_core(
                    marble,
                    num_new_neurons=int(n_neurons),
                    num_new_synapses=int(n_synapses),
                    alternative_connection_prob=float(alt_prob),
                    target_tier=tier or None,
                    neuron_types=types,
                )
                st.success("Core expanded")

            st.subheader("Add Neuron")
            n_type = st.text_input("Neuron Type", "standard", key="add_neuron_type")
            n_tier = st.text_input("Tier", "", key="add_neuron_tier")
            if st.button("Add Neuron", key="add_neuron_btn"):
                nid = add_neuron_to_marble(
                    marble, neuron_type=n_type, tier=n_tier or None
                )
                st.write(f"Neuron ID {nid} added")

            st.subheader("Add Synapse")
            src = st.number_input("Source ID", value=0, step=1, key="syn_src")
            tgt = st.number_input("Target ID", value=1, step=1, key="syn_tgt")
            weight = st.number_input("Weight", value=1.0, format="%f", key="syn_weight")
            stype = st.text_input("Synapse Type", "standard", key="syn_type")
            if st.button("Add Synapse", key="add_synapse_btn"):
                add_synapse_to_marble(
                    marble,
                    int(src),
                    int(tgt),
                    weight=float(weight),
                    synapse_type=stype,
                )
                st.success("Synapse added")

            st.subheader("Freeze Synapses")
            frac = st.number_input(
                "Fraction", min_value=0.0, max_value=1.0, value=0.5, key="freeze_frac"
            )
            if st.button("Freeze", key="freeze_btn"):
                freeze_synapses_fraction(marble, float(frac))
                st.success("Synapses frozen")

            st.subheader("Representation Size")
            inc = st.number_input("Increase By", value=1, step=1, key="inc_rep")
            if st.button("Increase Representation", key="inc_rep_btn"):
                increase_marble_representation(marble, delta=int(inc))
                st.success("Representation increased")
            dec = st.number_input("Decrease By", value=1, step=1, key="dec_rep")
            if st.button("Decrease Representation", key="dec_rep_btn"):
                decrease_marble_representation(marble, delta=int(dec))
                st.success("Representation decreased")

            st.subheader("Message Passing")
            mp_iter = st.number_input("Iterations", value=1, step=1, key="mp_iter")
            if st.button("Run Message Passing", key="mp_btn"):
                change = run_core_message_passing(marble, iterations=int(mp_iter))
                st.write(f"Avg change: {change}")

            st.subheader("Reset Representations")
            if st.button("Reset Representations", key="reset_reps"):
                reset_core_representations(marble)
                st.success("Representations reset")
            rand_std = st.number_input("Randomize STD", value=1.0, key="rand_std")
            if st.button("Randomize Representations", key="rand_reps"):
                randomize_core_representations(marble, std=float(rand_std))
                st.success("Representations randomized")

        with tab_cfg:
            st.write("Edit the active YAML configuration.")
            if "config_yaml" not in st.session_state:
                st.info("No configuration loaded")
            else:
                cfg_data = yaml.safe_load(st.session_state["config_yaml"]) or {}
                updated = render_config_editor(cfg_data)
                if st.button("Apply Changes", key="cfg_apply"):
                    st.session_state["config_yaml"] = yaml.safe_dump(
                        updated, sort_keys=False
                    )
                    st.success("Configuration updated")
                st.code(st.session_state["config_yaml"], language="yaml")
                if st.button("Reinitialize", key="cfg_reinit"):
                    st.session_state["marble"] = initialize_marble(
                        None, yaml_text=st.session_state["config_yaml"]
                    )
                    st.success("Reinitialized MARBLE")

        with tab_model:
            st.write("Convert a pretrained Hugging Face model into MARBLE.")
            model_query = st.text_input("Search Models", key="hf_model_query")
            if st.button("Search Models", key="hf_do_model_search") and model_query:
                st.session_state["hf_model_results"] = search_hf_models(model_query)
            if (
                "hf_model_results" in st.session_state
                and st.session_state["hf_model_results"]
            ):
                model_name = st.selectbox(
                    "Results",
                    st.session_state["hf_model_results"],
                    key="hf_model_choice",
                )
            else:
                model_name = st.text_input("Model Name", key="hf_model_name")
            if st.button("Preview Model", key="hf_preview") and model_name:
                try:
                    mdl = load_hf_model_wrapper(model_name)
                    st.session_state["hf_model"] = mdl
                    st.text(model_summary(mdl))
                except Exception as e:
                    st.error(str(e))

        with tab_offload:
            st.write(
                "Manage remote and torrent offloading for the current MARBLE instance."
            )
            if "remote_server" not in st.session_state:
                st.session_state["remote_server"] = None
            if "remote_client" not in st.session_state:
                st.session_state["remote_client"] = None
            if "torrent_tracker" not in st.session_state:
                st.session_state["torrent_tracker"] = None
            if "torrent_client" not in st.session_state:
                st.session_state["torrent_client"] = None

            with st.expander("Remote Server"):
                host = st.text_input("Host", "localhost", key="srv_host")
                port = st.number_input("Port", value=8000, step=1, key="srv_port")
                remote_url = st.text_input("Remote URL", key="srv_remote")
                if (
                    st.button("Start Server", key="srv_start")
                    and st.session_state["remote_server"] is None
                ):
                    st.session_state["remote_server"] = start_remote_server(
                        host=host,
                        port=int(port),
                        remote_url=remote_url or None,
                    )
                    st.success("Server started")
                if (
                    st.button("Stop Server", key="srv_stop")
                    and st.session_state["remote_server"] is not None
                ):
                    st.session_state["remote_server"].stop()
                    st.session_state["remote_server"] = None
                    st.success("Server stopped")

            with st.expander("Remote Client"):
                url = st.text_input(
                    "Server URL", "http://localhost:8000", key="cli_url"
                )
                if st.button("Create Client", key="cli_create"):
                    st.session_state["remote_client"] = create_remote_client(url)
                    st.success("Client created")
                if (
                    st.button("Attach to MARBLE", key="cli_attach")
                    and st.session_state.get("remote_client") is not None
                ):
                    marble.brain.remote_client = st.session_state["remote_client"]
                    marble.brain.offload_enabled = True
                    st.success("Client attached")

            with st.expander("Torrent Client"):
                client_id = st.text_input("Client ID", "main", key="tor_id")
                bufsize = st.number_input(
                    "Buffer Size", value=10, step=1, key="tor_buf"
                )
                beat = st.number_input("Heartbeat", value=30, step=1, key="tor_hb")
                if (
                    st.button("Start Torrent", key="tor_start")
                    and st.session_state["torrent_client"] is None
                ):
                    tracker, client = create_torrent_system(
                        client_id,
                        buffer_size=int(bufsize),
                        heartbeat_interval=int(beat),
                    )
                    st.session_state["torrent_tracker"] = tracker
                    st.session_state["torrent_client"] = client
                    marble.brain.torrent_client = client
                    marble.brain.torrent_offload_enabled = True
                    st.success("Torrent client started")
                if (
                    st.button("Stop Torrent", key="tor_stop")
                    and st.session_state["torrent_client"] is not None
                ):
                    st.session_state["torrent_client"].disconnect()
                    st.session_state["torrent_client"] = None
                    st.session_state["torrent_tracker"] = None
                    marble.brain.torrent_client = None
                    marble.brain.torrent_offload_enabled = False
                    st.success("Torrent client stopped")
            st.divider()
            if st.button("Offload High Attention", key="do_remote"):
                marble.brain.offload_high_attention(marble.brain.offload_threshold)
            if st.button("Offload via Torrent", key="do_torrent"):
                marble.brain.offload_high_attention_torrent(
                    marble.brain.torrent_offload_threshold
                )
            if st.button("Convert to MARBLE", key="hf_convert") and model_name:
                try:
                    marble = convert_hf_model(model_name)
                    st.session_state["marble"] = marble
                    st.success("Model converted")
                except Exception as e:
                    st.error(str(e))

        with tab_async:
            st.write("Control asynchronous training and auto-firing.")
            if st.button("Start Background Training", key="async_start"):
                if train_file is not None:
                    examples = load_examples(train_file)
                elif "hf_examples" in st.session_state:
                    examples = st.session_state["hf_examples"]
                else:
                    examples = []
                if examples:
                    start_background_training(marble, examples, epochs=epochs)
                    st.success("Training started")
                else:
                    st.error("No dataset loaded")
            if st.button("Wait For Training", key="async_wait"):
                wait_for_training(marble)
                st.success("Training complete")
            st.write(f"Training active: {training_in_progress(marble)}")
            if st.button("Start Auto-Firing", key="af_start"):
                start_auto_firing(marble)
                st.success("Auto-firing started")
            if st.button("Stop Auto-Firing", key="af_stop"):
                stop_auto_firing(marble)
                st.success("Auto-firing stopped")

        with tab_rl:
            st.write("Run reinforcement learning experiments in GridWorld.")
            size = st.number_input("Grid Size", min_value=2, value=4, step=1)
            episodes = st.number_input("Episodes", min_value=1, value=10, step=1)
            max_steps = st.number_input("Max Steps", min_value=1, value=50, step=1)
            double_q = st.checkbox("Double Q-learning")
            if st.button("Run GridWorld", key="run_gridworld"):
                rewards = run_gridworld_episode(
                    marble,
                    episodes=int(episodes),
                    max_steps=int(max_steps),
                    size=int(size),
                    double_q=double_q,
                )
                fig = go.Figure()
                fig.add_scatter(y=rewards, mode="lines+markers")
                fig.update_layout(xaxis_title="Episode", yaxis_title="Total Reward")
                st.plotly_chart(fig, use_container_width=True)

        with tab_adapt:
            st.write("Inspect adaptive controllers and dimensional tools.")
            info = meta_controller_info(marble)
            st.json(info)
            with st.expander("Update Meta Controller"):
                hlen = st.number_input(
                    "History Length",
                    value=info["history_length"],
                    step=1,
                    key="mc_hist",
                )
                adj = st.number_input(
                    "Adjustment",
                    value=info["adjustment"],
                    key="mc_adj",
                )
                min_t = st.number_input(
                    "Min Threshold",
                    value=info["min_threshold"],
                    key="mc_min",
                )
                max_t = st.number_input(
                    "Max Threshold",
                    value=info["max_threshold"],
                    key="mc_max",
                )
                if st.button("Apply Settings", key="mc_update"):
                    new = update_meta_controller(
                        marble,
                        history_length=int(hlen),
                        adjustment=float(adj),
                        min_threshold=float(min_t),
                        max_threshold=float(max_t),
                    )
                    st.json(new)
            if st.button("Adjust Now", key="mc_do_adjust"):
                thr = adjust_meta_controller(marble)
                st.success(f"Plasticity threshold = {thr}")
            if st.button("Clear History", key="mc_clear_hist"):
                reset_meta_loss_history(marble)
                st.success("History cleared")
            st.divider()
            se_hist = super_evo_history(marble)
            if se_hist:
                st.subheader("Super Evolution History")
                st.dataframe(pd.DataFrame(se_hist), use_container_width=True)
            se_changes = super_evo_changes(marble)
            if se_changes:
                st.subheader("Parameter Changes")
                st.dataframe(pd.DataFrame(se_changes), use_container_width=True)
                if st.button("Clear Change Log", key="se_clear"):
                    clear_super_evo_changes(marble)
                    st.success("Cleared change log")
            st.divider()
            if marble.get_brain().dim_search is not None:
                ds_loss = st.number_input(
                    "Dimensional Search Loss",
                    value=0.0,
                    key="ds_loss_val",
                )
                if st.button("Evaluate Dimensional Search", key="ds_eval"):
                    size = run_dimensional_search(marble, loss=ds_loss)
                    st.write(f"Representation size: {size}")
            if marble.get_brain().nd_topology is not None:
                nd_loss = st.number_input(
                    "N-D Topology Loss",
                    value=0.0,
                    key="nd_loss_val",
                )
                if st.button("Evaluate N-D Topology", key="nd_eval"):
                    size = run_nd_topology(marble, loss=nd_loss)
                    st.write(f"Representation size: {size}")

        with tab_memory:
            st.write("Interact with the Hybrid Memory system.")
            if marble.hybrid_memory is None:
                vec = st.text_input(
                    "Vector Store Path", "vector_store.pkl", key="hm_vec"
                )
                sym = st.text_input(
                    "Symbolic Store Path", "symbolic_memory.pkl", key="hm_sym"
                )
                if st.button("Create Hybrid Memory", key="hm_create"):
                    create_hybrid_memory(marble, vec, sym)
                    st.success("Hybrid memory initialized")
            else:
                key = st.text_input("Key", key="hm_key")
                val = st.text_input("Value", key="hm_val")
                if st.button("Store", key="hm_store") and key:
                    hybrid_memory_store(marble, key, val)
                    st.success("Stored")
                qval = st.text_input("Query", key="hm_query")
                topk = st.number_input(
                    "Top K", min_value=1, value=1, step=1, key="hm_top"
                )
                if st.button("Retrieve", key="hm_retrieve") and qval:
                    res = hybrid_memory_retrieve(marble, qval, top_k=int(topk))
                    st.write(res)
                maxe = st.number_input(
                    "Max Entries", min_value=1, value=1000, step=1, key="hm_max"
                )
                if st.button("Forget Old", key="hm_forget"):
                    hybrid_memory_forget(marble, max_entries=int(maxe))
                    st.success("Pruned old entries")

        with tab_nbexp:
            st.write("Explore Neuronenblitz wander behaviour.")
            nb_input = st.text_input("Input", value="0.0", key="nb_in")
            plast = st.checkbox("Apply Plasticity", value=True, key="nb_plast")
            procs = st.number_input(
                "Processes", min_value=1, value=1, step=1, key="nb_proc"
            )
            if st.button("Wander", key="nb_wander"):
                val = _parse_value(nb_input)
                out, ids = wander_neuronenblitz(marble, val, apply_plasticity=plast)
                st.write(f"Output: {out}")
                st.write(f"Path: {ids}")
            if st.button("Parallel Wander", key="nb_pwander"):
                val = _parse_value(nb_input)
                res = parallel_wander_neuronenblitz(marble, val, processes=int(procs))
                st.write(res)
            if st.button("Show Training History", key="nb_hist"):
                hist = marble.get_neuronenblitz().get_training_history()
                if hist:
                    st.dataframe(pd.DataFrame(hist), use_container_width=True)
                else:
                    st.info("No history available")

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

        with tab_tests:
            st.write("Run repository unit tests to verify functionality.")
            tests = list_test_files()
            selected = st.multiselect("Test Files", tests, key="test_select")
            pattern = (
                " or ".join(os.path.splitext(t)[0] for t in selected)
                if selected
                else None
            )
            if st.button("Run Tests", key="run_tests_btn"):
                output = run_tests(pattern)
                st.text(output if output else "No output")

        with tab_docs:
            st.write("View repository documentation.")
            docs = list_documentation_files()
            doc_choice = st.selectbox("Document", docs, key="doc_select")
            text = load_documentation(doc_choice)
            lang = "markdown" if doc_choice.endswith(".md") else "yaml"
            st.code(text, language=lang)

        with tab_browser:
            st.write("Browse dataset files and inspect samples.")
            file = st.file_uploader(
                "Dataset", type=["csv", "json", "jsonl", "zip"], key="ds_browser_file"
            )
            if file is not None:
                df = preview_file_dataset(file)
                st.dataframe(df.head(), use_container_width=True)
                idx = st.number_input("Sample index", min_value=0, max_value=len(df)-1, value=0, step=1, key="bit_idx")
                row = df.iloc[int(idx)]
                bits = flatten_tensor_to_bitstream(bytes_to_tensors(object_to_bytes(row["input"])))
                arr = np.array(bits, dtype=np.uint8).reshape(-1, 8) * 255
                st.image(arr, caption="Input Bits", width=200)

        with tab_src:
            st.write("Browse repository source code.")
            modules = list_repo_modules()
            mod_choice = st.selectbox("Module", modules, key="src_mod")
            if st.button("Show Source", key="src_show"):
                code = load_module_source(mod_choice)
                st.code(code, language="python")

    _persist_ui_state()


if __name__ == "__main__":
    run_playground()


def activation_figure(
    core, activations: dict[int, float], layout: str = "spring"
) -> go.Figure:
    """Return a Plotly figure showing neuron activations."""
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
    node_x, node_y, act_vals = [], [], []
    for n in g.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        act_vals.append(float(activations.get(n, 0.0)))
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
        marker=dict(
            size=6, color=act_vals, colorscale="Viridis", colorbar=dict(title="Act")
        ),
    )
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
    return fig
