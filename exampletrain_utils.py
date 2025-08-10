"""Utility helpers for ``exampletrain`` tests.

This module provides lightweight fixtures that allow the heavy
``exampletrain.py`` script to be tested without downloading large
datasets or initialising the full Stable Diffusion pipeline.  The
utilities here are intentionally self‑contained and operate entirely on
CPU or GPU depending on availability so that parity between devices can
be verified easily.

Currently the module exposes :func:`create_synthetic_dataset` which
produces a deterministic sequence of input/target pairs.  Each value is
generated using the supplied random seed so tests can rely on
reproducible behaviour.  When a CUDA device is available the tensors are
allocated on that device; otherwise they default to the CPU.  Callers
can also explicitly select the device.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch


def create_synthetic_dataset(
    num_samples: int = 10,
    device: Optional[str] = None,
    seed: int = 0,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Generate a deterministic synthetic dataset.

    Parameters
    ----------
    num_samples:
        Number of ``(input, target)`` pairs to produce.
    device:
        The device on which tensors should be allocated.  If ``None`` the
        function will automatically select ``"cuda"`` when available or
        fallback to ``"cpu"`` otherwise.
    seed:
        Random seed controlling the generated values.

    Returns
    -------
    list of tuple of ``torch.Tensor``
        Each tuple contains an ``input`` tensor and a corresponding
        ``target`` tensor.  The relationship ``target = 2 * input + 1`` is
        intentionally simple yet non‑trivial, enabling tests to verify
        training and inference logic without relying on external
        resources.
    """

    actual_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=actual_device)
    gen.manual_seed(seed)

    inputs = torch.rand(num_samples, generator=gen, device=actual_device)
    targets = 2 * inputs + 1

    return list(zip(inputs, targets))


class MockTokenizer:
    """Deterministic tokenizer returning a fixed sequence of token IDs.

    The tokenizer mirrors the interface of the Hugging Face tokenizer used
    by the original ``exampletrain.py`` script.  Regardless of the provided
    text, it returns the same token ID tensor so that tests can exercise the
    downstream components without relying on external models.  The returned
    tensor lives on the device specified at construction time, defaulting to
    CUDA when available.
    """

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, text: str, return_tensors: str = "pt") -> dict:
        if return_tensors != "pt":  # pragma: no cover - defensive programming
            raise ValueError("MockTokenizer only supports return_tensors='pt'")
        token_ids = torch.tensor([[1, 2, 3]], device=self.device)
        return {"input_ids": token_ids}


class MockTextEncoder(torch.nn.Module):
    """Minimal text encoder producing deterministic embeddings.

    The encoder consists of an identity linear layer so that the output
    matches the input token IDs (cast to ``float``).  This behaviour is
    sufficient for parity tests while remaining device‑agnostic.
    """

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__()
        actual_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.linear = torch.nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(3))
        self.to(actual_device)

    def forward(self, input_ids: torch.Tensor) -> "MockEncoderOutput":
        embeddings = self.linear(input_ids.float())
        return MockEncoderOutput(embeddings)


class MockEncoderOutput:
    """Container mimicking the output of a Hugging Face text encoder."""

    def __init__(self, last_hidden_state: torch.Tensor) -> None:
        self.last_hidden_state = last_hidden_state


class MockStableDiffusionPipeline:
    """Lightweight stand‑in for ``StableDiffusionPipeline`` used in tests.

    The pipeline exposes a :attr:`tokenizer` and :attr:`text_encoder` that
    operate purely in memory without external resources.  The :meth:`to`
    method mirrors the behaviour of the real pipeline, allowing tests to
    switch devices seamlessly.
    """

    def __init__(self, device: Optional[str] = None) -> None:
        actual_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = MockTokenizer(device=actual_device)
        self.text_encoder = MockTextEncoder(device=actual_device)

    def to(self, device: str) -> "MockStableDiffusionPipeline":
        self.tokenizer.device = device
        self.text_encoder.to(device)
        return self


__all__ = [
    "create_synthetic_dataset",
    "MockTokenizer",
    "MockTextEncoder",
    "MockStableDiffusionPipeline",
]
