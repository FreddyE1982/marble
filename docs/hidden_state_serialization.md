# Hidden State Serialization

The RNN converter now stores initial hidden states alongside layer metadata.
Each entry records the layer index, direction, tensor values, shape, dtype,
and device. The information is serialised as JSON through `core_to_json` so it
remains portable across CPU and GPU environments.
