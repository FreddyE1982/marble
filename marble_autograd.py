import torch
import torch.nn as nn
from marble_brain import Brain

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
        return None, grad_output.clone()

class MarbleAutogradLayer(nn.Module):
    """Transparent autograd wrapper for a :class:`Brain` instance."""

    def __init__(self, brain: Brain, learning_rate: float = 0.01, accumulation_steps: int = 1) -> None:
        super().__init__()
        self.brain = brain
        self.learning_rate = learning_rate
        self.accumulation_steps = max(1, int(accumulation_steps))
        self._grad_buffer: dict = {}
        self._accum_counter = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return MarbleAutogradFunction.apply(self, x)
