# Hidden State Serialization

The RNN converter now stores initial hidden states alongside layer metadata.
Each entry records the layer index, direction, tensor values, shape, dtype,
and device. The information is serialised as JSON through `core_to_json` so it
remains portable across CPU and GPU environments.

When a model is loaded via `core_from_json`, these serialized values are
automatically mapped back to their corresponding RNN neurons.  The helper
`restore_hidden_states` assigns each value to the neuron matching the recorded
layer index and direction, emitting a warning if the stored tensor length does
not match the expected hidden size.  Device information ensures that restored
states live on the correct CPU or CUDA backend, preserving training context.
