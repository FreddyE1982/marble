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

For layers that are not yet implemented you can explicitly mark them as
unsupported using ``unsupported_layer``. This registers a stub converter that
raises :class:`UnsupportedLayerError` with a standardised message:

```python
from pytorch_to_marble import unsupported_layer

unsupported_layer(torch.nn.MaxPool3d)
```

A complete working example demonstrating how to register and use a custom
converter can be found in
``examples/custom_converter_example.py``.  The script defines a
``DoubleLinear`` module, registers a converter with
``@register_converter`` and converts the model on CPU or GPU depending on
availability.

When ``convert_model`` encounters ``MaxPool3d`` it will now immediately raise a
clear error indicating the layer is not supported. This makes gaps in converter
coverage visible while providing guidance for future contributors.
The shipped registry covers a wide range of core building blocks including:

- Linear, Conv2d and the activations ReLU, Sigmoid, Tanh and GELU
- Dropout, Flatten and Unflatten for shape manipulation
- MaxPool2d, AvgPool2d and their adaptive and global variants
- Embedding and EmbeddingBag layers with ``padding_idx`` and ``max_norm``
- Recurrent modules RNN, LSTM and GRU
- BatchNorm1d/2d, LayerNorm and GroupNorm
- Sequential containers and ModuleList objects which expand recursively

Pooling layers like ``MaxPool2d`` and ``AvgPool2d`` are handled by dedicated
converters that create neurons with ``neuron_type`` set to ``"maxpool2d"`` or
``"avgpool2d"`` and store kernel parameters in ``neuron.params``.

Normalization layers ``LayerNorm`` and ``GroupNorm`` mark their input neurons
with ``neuron_type`` set to ``"layernorm"`` or ``"groupnorm"`` respectively.
Important parameters such as ``normalized_shape`` or ``num_groups`` as well as
``eps`` are stored in ``neuron.params`` for use during message passing.
