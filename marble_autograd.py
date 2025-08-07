import torch
import torch.nn as nn
from marble_brain import Brain
from typing import Callable, Optional

class MarbleAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, wrapper, input_tensor: torch.Tensor) -> torch.Tensor:
        device = input_tensor.device
        flat_input = input_tensor.reshape(-1)
        outputs = []
        paths = []
        for val in flat_input:
            scalar = val.detach().cpu().numpy().reshape(-1)[0]
            out, path = wrapper.brain.neuronenblitz.dynamic_wander(scalar)
            outputs.append(out)
            paths.append(path)
        ctx.wrapper = wrapper
        ctx.paths = paths
        ctx.save_for_backward(input_tensor)
        output_tensor = torch.tensor(outputs, dtype=input_tensor.dtype, device=device)
        return output_tensor.reshape_as(input_tensor)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input_tensor, = ctx.saved_tensors
        wrapper = ctx.wrapper
        flat_grads = grad_output.reshape(-1)
        for path, grad_out in zip(ctx.paths, flat_grads):
            grad_val = grad_out.detach().cpu().numpy().reshape(-1)[0]
            for syn in path:
                source_val = wrapper.brain.core.neurons[syn.source].value
                if source_val is None:
                    source_val = 0.0
                grad = grad_val * source_val
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
    """Insert MARBLE into a PyTorch model and optionally mix its output."""

    def __init__(self, brain: Brain, train_in_graph: bool = True, mix_weight: float = 0.0) -> None:
        super().__init__()
        self.marble_layer = MarbleAutogradLayer(brain)
        self.train_in_graph = train_in_graph
        self.mix_weight = mix_weight

    def forward(self, x: torch.Tensor, mix_weight: Optional[float] = None) -> torch.Tensor:
        weight = self.mix_weight if mix_weight is None else mix_weight
        if self.train_in_graph:
            marble_out = self.marble_layer(x)
        else:
            with torch.no_grad():
                marble_out = self.marble_layer(x)
        return x + marble_out * weight

    def train_marble(self, examples, epochs: int = 1) -> None:
        self.marble_layer.brain.train(examples, epochs=epochs)

    def infer_marble(self, inp: float) -> float:
        return self.marble_layer.brain.infer(inp)
