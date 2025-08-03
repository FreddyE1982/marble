import torch
from dataclasses import dataclass
from typing import Tuple, Dict, Any


@dataclass
class QuantizedTensor:
    """Represents a uniformly quantized tensor with optional bit packing.

    Parameters
    ----------
    bits: torch.ByteTensor
        Packed representation of quantized values.
    shape: Tuple[int, ...]
        Original tensor shape.
    scale: float
        Quantization scale factor.
    zero_point: int
        Zero point used during quantization.
    bit_width: int
        Number of bits per quantized value (1-8).
    device: torch.device
        Device on which the tensor resides.
    """

    bits: torch.ByteTensor
    shape: Tuple[int, ...]
    scale: float
    zero_point: int
    bit_width: int
    device: torch.device

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, bit_width: int = 8) -> "QuantizedTensor":
        """Quantize a dense tensor into a :class:`QuantizedTensor`.

        Parameters
        ----------
        tensor: torch.Tensor
            Input tensor to quantize. Must be floating point.
        bit_width: int, optional
            Number of bits for quantization. Supports values from 1 to 8.
        """
        if bit_width < 1 or bit_width > 8:
            raise ValueError("bit_width must be between 1 and 8")
        if not tensor.is_floating_point():
            raise TypeError("tensor must be floating point")

        device = tensor.device
        qmin = 0
        qmax = 2 ** bit_width - 1
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        if abs(max_val - min_val) < 1e-8:
            scale = 1.0
        else:
            scale = (max_val - min_val) / float(qmax - qmin)
        zero_point = int(round(qmin - min_val / scale))
        q_tensor = ((tensor / scale) + zero_point).clamp(qmin, qmax).round().to(torch.int32)
        bits = cls._pack_bits(q_tensor, bit_width)
        return cls(bits=bits, shape=tuple(tensor.shape), scale=float(scale),
                   zero_point=zero_point, bit_width=bit_width, device=device)

    def to_dense(self) -> torch.Tensor:
        """Dequantize into a dense floating point tensor."""
        q_tensor = self._unpack_bits(self.bits, self.bit_width, self.shape, self.device)
        return (q_tensor.to(torch.float32) - self.zero_point) * self.scale

    def to_bits(self) -> torch.ByteTensor:
        """Return the packed bit representation."""
        return self.bits

    def state_dict(self) -> Dict[str, Any]:
        """Serialize tensor to a Python dict."""
        return {
            "bits": self.bits.cpu(),
            "shape": self.shape,
            "scale": self.scale,
            "zero_point": self.zero_point,
            "bit_width": self.bit_width,
            "device": str(self.device),
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "QuantizedTensor":
        """Deserialize from a state dict."""
        device = torch.device(state["device"])
        bits = state["bits"].to(device)
        return cls(bits=bits, shape=tuple(state["shape"]), scale=float(state["scale"]),
                   zero_point=int(state["zero_point"]), bit_width=int(state["bit_width"]),
                   device=device)

    @staticmethod
    def _pack_bits(q_tensor: torch.Tensor, bit_width: int) -> torch.ByteTensor:
        """Pack quantized integers into a byte tensor."""
        q_tensor = q_tensor.to(torch.int32)
        device = q_tensor.device
        if bit_width == 8:
            return q_tensor.to(torch.uint8).contiguous()
        values_per_byte = 8 // bit_width
        num_vals = q_tensor.numel()
        padded_len = ((num_vals + values_per_byte - 1) // values_per_byte) * values_per_byte
        padded = torch.zeros(padded_len, dtype=torch.int32, device=device)
        padded[:num_vals] = q_tensor.view(-1)
        padded = padded.view(-1, values_per_byte)
        shifts = torch.arange(values_per_byte, device=device, dtype=torch.int32) * bit_width
        packed = (padded << shifts).sum(dim=1)
        return packed.to(torch.uint8)

    @staticmethod
    def _unpack_bits(bits: torch.ByteTensor, bit_width: int, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Unpack byte tensor into quantized integers."""
        bits = bits.to(torch.int32).to(device)
        if bit_width == 8:
            q = bits
        else:
            values_per_byte = 8 // bit_width
            shifts = torch.arange(values_per_byte, device=device, dtype=torch.int32) * bit_width
            mask = (1 << bit_width) - 1
            expanded = ((bits.unsqueeze(1) >> shifts) & mask).reshape(-1)
            q = expanded
        q = q[:int(torch.prod(torch.tensor(shape)))]
        return q.view(*shape)


__all__ = ["QuantizedTensor"]

