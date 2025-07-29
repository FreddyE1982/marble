# PyTorch Conversion Registry

MARBLE's model converter relies on a registry that maps PyTorch layers and
operations to conversion functions. Each converter receives the layer instance,
the active :class:`marble_core.Core` and the input neuron IDs. It returns the
IDs of the output neurons created in the MARBLE graph.

Converters are registered with decorators:

```python
from pytorch_to_marble import register_converter

@register_converter(torch.nn.Linear)
def convert_linear(layer, core, inputs):
    ...
```

The registry dictionaries ``LAYER_CONVERTERS``, ``FUNCTION_CONVERTERS`` and
``METHOD_CONVERTERS`` hold these mappings. Custom layers can be supported by
adding new entries using ``register_converter``.

Pooling layers like ``MaxPool2d`` and ``AvgPool2d`` are handled by dedicated
converters that create neurons with ``neuron_type`` set to ``"maxpool2d"`` or
``"avgpool2d"`` and store kernel parameters in ``neuron.params``.
