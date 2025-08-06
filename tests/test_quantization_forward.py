import torch
from quantized_tensor import QuantizedTensor


def test_linear_forward_close():
    lin = torch.nn.Linear(8, 4, bias=False)
    x = torch.randn(2, 8)
    qt = QuantizedTensor.from_tensor(lin.weight.data, bit_width=4)
    lin_q = torch.nn.Linear(8, 4, bias=False)
    lin_q.weight.data = qt.to_dense()
    out_dense = lin(x)
    out_quant = lin_q(x)
    assert torch.allclose(out_dense, out_quant, atol=qt.scale * 4)
