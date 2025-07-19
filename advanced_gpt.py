"""Lightweight transformer decoder with optional CuPy acceleration."""

from __future__ import annotations

import os
from typing import List, Tuple, Dict

try:
    import cupy as cp

    if cp.cuda.runtime.getDeviceCount() > 0:
        xp = cp
    else:
        raise Exception
except Exception:  # pragma: no cover - fallback
    import numpy as cp
    xp = cp


def unbroadcast(grad: xp.ndarray, shape: Tuple[int, ...]) -> xp.ndarray:
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    def __init__(self, data: xp.ndarray, parents: Tuple["Tensor", ...] = (), op: str | None = None) -> None:
        self.data = xp.array(data, dtype=xp.float32)
        self.grad = xp.zeros_like(self.data)
        self.parents = parents
        self.op = op
        self._backward = lambda: None

    def __add__(self, other: "Tensor") -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "add")

        def _backward() -> None:
            self.grad += unbroadcast(out.grad, self.data.shape)
            other.grad += unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other: "Tensor") -> "Tensor":
        out = Tensor(self.data @ other.data, (self, other), "matmul")

        def _backward() -> None:
            self.grad += out.grad @ xp.swapaxes(other.data, -1, -2)
            other.grad += xp.swapaxes(self.data, -1, -2) @ out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: "Tensor") -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "mul")

        def _backward() -> None:
            self.grad += unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += unbroadcast(out.grad * self.data, other.data.shape)

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        out = Tensor(xp.tanh(self.data), (self,), "tanh")

        def _backward() -> None:
            self.grad += (1 - out.data ** 2) * out.grad

        out._backward = _backward
        return out

    def reshape(self, *shape: int) -> "Tensor":
        out = Tensor(self.data.reshape(*shape), (self,), "reshape")

        def _backward() -> None:
            self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def transpose(self, axes: Tuple[int, ...]) -> "Tensor":
        out = Tensor(self.data.transpose(axes), (self,), "transpose")

        def _backward() -> None:
            inv = xp.argsort(xp.array(axes))
            self.grad += out.grad.transpose(inv)

        out._backward = _backward
        return out

    def backward(self, grad: xp.ndarray | None = None) -> None:
        if grad is None:
            grad = xp.ones_like(self.data)
        self.grad = grad
        topo: List[Tensor] = []
        visited = set()

        def build(v: Tensor) -> None:
            if id(v) not in visited:
                visited.add(id(v))
                for child in v.parents:
                    build(child)
                topo.append(v)

        build(self)
        for v in reversed(topo):
            v._backward()


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    e = xp.exp(x.data - xp.max(x.data, axis=axis, keepdims=True))
    out = Tensor(e / e.sum(axis=axis, keepdims=True), (x,), "softmax")

    def _backward() -> None:
        grad = out.grad
        s = out.data
        dx = (grad - (grad * s).sum(axis=axis, keepdims=True)) * s
        x.grad += dx

    out._backward = _backward
    return out


def cross_entropy(logits: Tensor, targets: xp.ndarray) -> Tensor:
    probs = softmax(logits, axis=-1)
    N = targets.shape[0]
    loss_val = -xp.log(probs.data[xp.arange(N), targets]).mean()
    out = Tensor(loss_val, (logits,), "cross_entropy")

    def _backward() -> None:
        grad = probs.data
        grad[xp.arange(N), targets] -= 1
        grad /= N
        logits.grad += grad * out.grad

    out._backward = _backward
    return out


def embed(weight: Tensor, idx: xp.ndarray) -> Tensor:
    out = Tensor(weight.data[idx], (weight,), "embed")

    def _backward() -> None:
        xp.add.at(weight.grad, idx, out.grad)

    out._backward = _backward
    return out


def load_text_dataset(path: str, vocab_size: int, block_size: int) -> Tuple[List[xp.ndarray], Dict[str, int]]:
    """Read a text file and convert it into token sequences."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    unique = sorted(list(set(text)))[:vocab_size]
    vocab = {ch: i for i, ch in enumerate(unique)}
    data: List[xp.ndarray] = []
    for i in range(0, len(text) - block_size):
        seq = [vocab.get(ch, 0) for ch in text[i : i + block_size + 1]]
        data.append(xp.array(seq, dtype=xp.int32))
    return data, vocab


class AdvancedGPT:
    """GPT model using custom autograd on NumPy/CuPy."""

    def __init__(self, vocab_size: int, block_size: int, num_layers: int = 2, num_heads: int = 2, hidden_dim: int = 64) -> None:
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        scale = 0.02

        self.embed = Tensor(xp.random.randn(vocab_size, hidden_dim) * scale)
        self.pos_embed = Tensor(xp.random.randn(block_size, hidden_dim) * scale)
        self.layers = []
        for _ in range(num_layers):
            layer = {
                "wq": Tensor(xp.random.randn(hidden_dim, hidden_dim) * scale),
                "wk": Tensor(xp.random.randn(hidden_dim, hidden_dim) * scale),
                "wv": Tensor(xp.random.randn(hidden_dim, hidden_dim) * scale),
                "wo": Tensor(xp.random.randn(hidden_dim, hidden_dim) * scale),
                "w1": Tensor(xp.random.randn(hidden_dim, 4 * hidden_dim) * scale),
                "w2": Tensor(xp.random.randn(4 * hidden_dim, hidden_dim) * scale),
            }
            self.layers.append(layer)
        self.lm_head = Tensor(xp.random.randn(hidden_dim, vocab_size) * scale)

    def parameters(self) -> List[Tensor]:
        params = [self.embed, self.pos_embed, self.lm_head]
        for layer in self.layers:
            params.extend([layer["wq"], layer["wk"], layer["wv"], layer["wo"], layer["w1"], layer["w2"]])
        return params

    def __call__(self, idx: xp.ndarray) -> Tensor:
        T = len(idx)
        x = embed(self.embed, idx) + embed(self.pos_embed, xp.arange(T))
        for layer in self.layers:
            q = x @ layer["wq"]
            k = x @ layer["wk"]
            v = x @ layer["wv"]

            q = q.reshape(T, self.num_heads, self.head_dim).transpose((1, 0, 2))
            k = k.reshape(T, self.num_heads, self.head_dim).transpose((1, 2, 0))
            v = v.reshape(T, self.num_heads, self.head_dim).transpose((1, 0, 2))

            scores = (q @ k) * (1.0 / xp.sqrt(self.head_dim))
            mask = xp.tril(xp.ones((T, T), dtype=bool))
            scores_data = scores.data.copy()
            scores_data[:, ~mask] = -1e9
            masked = Tensor(scores_data, (scores,), "mask")

            def _backward() -> None:
                scores.grad += xp.where(mask, masked.grad, 0)

            masked._backward = _backward
            scores = masked

            weights = softmax(scores, axis=-1)
            attn = weights @ v
            attn = attn.transpose((1, 0, 2)).reshape(T, self.hidden_dim)
            x = attn @ layer["wo"]
            ff = (x @ layer["w1"]).tanh()
            x = x + ff @ layer["w2"]
        logits = x @ self.lm_head
        return logits


def train_advanced_gpt(
    dataset: List[xp.ndarray],
    vocab_size: int,
    block_size: int,
    num_layers: int = 2,
    num_heads: int = 2,
    hidden_dim: int = 64,
    epochs: int = 1,
    lr: float = 1e-3,
    batch_size: int = 1,
    seed: int | None = None,
) -> Tuple[AdvancedGPT, List[float]]:
    if seed is not None:
        xp.random.seed(seed)
    model = AdvancedGPT(vocab_size, block_size, num_layers, num_heads, hidden_dim)
    losses: List[float] = []
    for _ in range(epochs):
        total = 0.0
        for start in range(0, len(dataset), batch_size):
            batch = dataset[start : start + batch_size]
            for seq in batch:
                inp = seq[:-1]
                target = seq[1:]
                logits = model(inp)
                loss = cross_entropy(logits, target)
                loss.backward()
                for p in model.parameters():
                    p.data -= lr * p.grad
                    p.grad = xp.zeros_like(p.grad)
                total += float(loss.data)
        losses.append(total / max(len(dataset), 1))
    return model, losses


__all__ = [
    "xp",
    "load_text_dataset",
    "AdvancedGPT",
    "train_advanced_gpt",
]
