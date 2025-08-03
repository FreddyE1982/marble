# Graph Builder Utilities

The graph builder helpers simplify constructing MARBLE cores programmatically. Key utilities include:

- `add_fully_connected_layer`: creates neuron groups with specified activation functions and connects them with weighted synapses.
- `add_convolutional_layer`: builds multi-channel convolutional structures, mapping kernels to synapse weights automatically.
- `add_pooling_layer`: supports max and average pooling by wiring reduction neurons.

These helpers operate on a :class:`~marble_core.Core` instance and return the IDs of output neurons. They ensure that neuron and synapse counts update consistently for CPU and GPU execution.
