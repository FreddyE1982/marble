import torch
import torch.nn as nn
from marble_brain import Brain
from typing import Callable, Optional

class MarbleAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, wrapper, input_tensor: torch.Tensor) -> torch.Tensor:
        input_value = float(input_tensor.item())
        output, path = wrapper.brain.neuronenblitz.dynamic_wander(input_value)
        ctx.wrapper = wrapper
        ctx.path = path
        ctx.save_for_backward(input_tensor)
        return torch.tensor(output, dtype=input_tensor.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input_tensor, = ctx.saved_tensors
        wrapper = ctx.wrapper
        for syn in ctx.path:
            source_val = wrapper.brain.core.neurons[syn.source].value
            if source_val is None:
                source_val = 0.0
            grad = float(grad_output) * source_val
            wrapper._grad_buffer[syn] = wrapper._grad_buffer.get(syn, 0.0) + grad
        wrapper._accum_counter += 1
        if wrapper._accum_counter >= wrapper.accumulation_steps:
            for syn, total in wrapper._grad_buffer.items():
                syn.weight -= wrapper.learning_rate * (total / wrapper.accumulation_steps)
            wrapper._grad_buffer.clear()
            wrapper._accum_counter = 0
            wrapper._step += 1
            if wrapper.scheduler:
                wrapper.learning_rate = wrapper.scheduler(wrapper._step)
        return None, grad_output.clone()

class MarbleAutogradLayer(nn.Module):
    """Transparent autograd wrapper for a :class:`Brain` instance."""

    def __init__(
        self,
        brain: Brain,
        learning_rate: float = 0.01,
        accumulation_steps: int = 1,
        scheduler: Optional[Callable[[int], float]] = None,
    ) -> None:
        super().__init__()
        self.brain = brain
        self.learning_rate = learning_rate
        self.accumulation_steps = max(1, int(accumulation_steps))
        self.scheduler = scheduler
        self._step = 0
        self._grad_buffer: dict = {}
        self._accum_counter = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return MarbleAutogradFunction.apply(self, x)

class TransparentMarbleLayer(nn.Module):
    """Insert MARBLE into a PyTorch model without altering activations."""

    def __init__(self, brain: Brain, train_in_graph: bool = True) -> None:
        super().__init__()
        self.marble_layer = MarbleAutogradLayer(brain)
        self.train_in_graph = train_in_graph

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.train_in_graph:
            marble_out = self.marble_layer(x)
            return x + marble_out * 0
        with torch.no_grad():
            self.marble_layer(x)
        return x

    def train_marble(self, examples, epochs: int = 1) -> None:
        self.marble_layer.brain.train(examples, epochs=epochs)

    def infer_marble(self, inp: float) -> float:
        return self.marble_layer.brain.infer(inp)
