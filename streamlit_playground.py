from __future__ import annotations

import os
import json
import wave
import tempfile
from io import BytesIO
import io
import runpy
import contextlib
import glob

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
    expand_marble_core,
    add_neuron_to_marble,
    add_synapse_to_marble,
    freeze_synapses_fraction,
    increase_marble_representation,
    decrease_marble_representation,
    run_core_message_passing,
    reset_core_representations,
    randomize_core_representations,
)

from metrics_dashboard import MetricsDashboard
import pkgutil
import importlib
import yaml
import torch
from transformers import AutoModel
import pytest


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


def search_hf_datasets(query: str, limit: int = 20) -> list[str]:
    """Return dataset IDs from the Hugging Face Hub matching ``query``."""
    from huggingface_hub import HfApi

    api = HfApi()
    datasets = api.list_datasets(search=query, limit=limit)
    return [d.id for d in datasets]


def search_hf_models(query: str, limit: int = 20) -> list[str]:
    """Return model IDs from the Hugging Face Hub matching ``query``."""
    from huggingface_hub import HfApi

    api = HfApi()
    models = api.list_models(search=query, limit=limit)
    return [m.id for m in models]


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
    from system_metrics import get_system_memory_usage, get_gpu_memory_usage

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


def start_remote_server(
    host: str = "localhost",
    port: int = 8000,
    remote_url: str | None = None,
    compression_level: int = 6,
    compression_enabled: bool = True,
) -> object:
    """Start and return a ``RemoteBrainServer`` instance."""
    from remote_offload import RemoteBrainServer

    server = RemoteBrainServer(
        host=host,
        port=port,
        remote_url=remote_url,
        compression_level=compression_level,
        compression_enabled=compression_enabled,
    )
    server.start()
    return server


def create_remote_client(
    url: str,
    timeout: float = 5.0,
    max_retries: int = 3,
    compression_level: int = 6,
    compression_enabled: bool = True,
) -> object:
    """Return a configured ``RemoteBrainClient``."""
    from remote_offload import RemoteBrainClient

    return RemoteBrainClient(
        url,
        timeout=timeout,
        max_retries=max_retries,
        compression_level=compression_level,
        compression_enabled=compression_enabled,
    )


def create_torrent_system(
    client_id: str = "main",
    buffer_size: int = 10,
    heartbeat_interval: int = 30,
) -> tuple[object, object]:
    """Return a tracker and connected ``BrainTorrentClient``."""
    from torrent_offload import BrainTorrentTracker, BrainTorrentClient

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
            or mod.name in {"streamlit_playground", "tests", "examples"}
        ):
            continue
        module = importlib.import_module(mod.name)
        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__name__.endswith("Learner") or obj.__name__.endswith("Agent"):
                modules.add(mod.name)
                break
    return sorted(modules)


def list_learner_classes(module_name: str) -> list[str]:
    """Return learner class names in ``module_name``."""
    module = importlib.import_module(module_name)
    classes = []
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if name.endswith("Learner") or name.endswith("Agent"):
            classes.append(name)
    return sorted(classes)


def create_learner(module_name: str, class_name: str, marble, **params):
    """Instantiate a learner class using components from ``marble``."""
    module = importlib.import_module(module_name)
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
        (
            tab_iface,
            tab_mod,
            tab_learner,
            tab_pipe,
            tab_code,
            tab_vis,
            tab_metrics,
            tab_stats,
            tab_core,
            tab_cfg,
            tab_model,
            tab_offload,
            tab_proj,
            tab_tests,
            tab_docs,
        ) = st.tabs(
            [
                "marble_interface",
                "Modules",
                "Learners",
                "Pipeline",
                "Custom Code",
                "Visualization",
                "Metrics",
                "System Stats",
                "Core Tools",
                "Config Editor",
                "Model Conversion",
                "Offloading",
                "Projects",
                "Tests",
                "Documentation",
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
                        st.experimental_rerun()
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
            if st.button("Run Pipeline") and st.session_state["pipeline"]:
                res = execute_function_sequence(st.session_state["pipeline"], marble)
                for out in res:
                    st.write(out)
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
            if st.button("Generate Graph", key="show_graph"):
                fig = core_figure(marble.get_core())
                st.plotly_chart(fig, use_container_width=True)

        with tab_metrics:
            st.write("Live metrics from the current run.")
            fig = metrics_figure(marble)
            st.plotly_chart(fig, use_container_width=True)
            st.button("Refresh", key="metrics_refresh")

        with tab_stats:
            st.write("System resource usage in megabytes.")
            stats = system_stats()
            st.metric("RAM", f"{stats['ram_mb']:.1f} MB")
            st.metric("GPU", f"{stats['gpu_mb']:.1f} MB")
            if st.button("Refresh Stats", key="stats_refresh"):
                st.experimental_rerun()

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
                    mdl = load_hf_model(model_name)
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
            doc_choice = st.selectbox(
                "Document", ["README", "TUTORIAL", "YAML Manual"], key="doc_select"
            )
            if doc_choice == "README":
                st.code(load_readme(), language="markdown")
            elif doc_choice == "TUTORIAL":
                st.code(load_tutorial(), language="markdown")
            else:
                st.code(load_yaml_manual(), language="yaml")


if __name__ == "__main__":
    run_playground()
