import argparse
from pathlib import Path
from string import Template
import torch
from typing import Dict, Any

TEMPLATES: Dict[str, Dict[str, Any]] = {
    "classification": {
        "description": "Synthetic classification pipeline with training loop",
        "parameters": {
            "input_dim": 4,
            "num_classes": 2,
            "epochs": 1,
        },
        "code": Template(
            """import argparse
import torch
from pipeline import Pipeline


def load_data(device: torch.device):
    x = torch.randn(32, ${input_dim}, device=device)
    y = torch.randint(0, ${num_classes}, (32,), device=device)
    return x, y


def train_step(device: torch.device, epochs: int = ${epochs}):
    model = torch.nn.Linear(${input_dim}, ${num_classes}).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    x, y = load_data(device)
    for _ in range(epochs):
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
    return float(loss.item())


def build_pipeline(device: torch.device) -> Pipeline:
    pipe = Pipeline()
    pipe.add_step("train_step", module=__name__, params={"device": device})
    return pipe


def main(device: torch.device) -> None:
    pipeline = build_pipeline(device)
    pipeline.execute()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    args = parser.parse_args()
    dev = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    main(dev)
"""
        ),
    },
    "preprocessing": {
        "description": "Data normalisation pipeline",
        "parameters": {
            "input_dim": 8,
        },
        "code": Template(
            """import argparse
import torch
from pipeline import Pipeline


def generate(device: torch.device):
    data = torch.randn(16, ${input_dim}, device=device)
    return data


def normalize(data: torch.Tensor) -> torch.Tensor:
    return (data - data.mean()) / data.std()


def build_pipeline(device: torch.device) -> Pipeline:
    pipe = Pipeline()
    pipe.add_step("generate", module=__name__, params={"device": device})
    pipe.add_step("normalize", module=__name__, depends_on=["generate"])
    return pipe


def main(device: torch.device) -> None:
    pipeline = build_pipeline(device)
    pipeline.execute()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    args = parser.parse_args()
    dev = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    main(dev)
"""
        ),
    },
}


def list_templates() -> Dict[str, str]:
    return {name: info["description"] for name, info in TEMPLATES.items()}


def generate_template(name: str, output: Path, **overrides: Any) -> Path:
    if name not in TEMPLATES:
        raise ValueError(f"Unknown template '{name}'")
    info = TEMPLATES[name]
    params = info["parameters"].copy()
    params.update(overrides)
    code = info["code"].substitute(**params)
    output.write_text(code)
    return output


def cli() -> None:
    parser = argparse.ArgumentParser(description="Generate pipeline templates")
    parser.add_argument("template", nargs="?", help="Template name")
    parser.add_argument("output", nargs="?", help="Output file")
    parser.add_argument("--list", action="store_true", help="List templates")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    args = parser.parse_args()

    if args.list:
        for name, desc in list_templates().items():
            print(f"{name}: {desc}")
        return
    if not args.template or not args.output:
        parser.error("template and output are required unless --list is used")
    overrides = {"device": args.device} if args.device else {}
    generate_template(args.template, Path(args.output), **overrides)


if __name__ == "__main__":
    cli()
