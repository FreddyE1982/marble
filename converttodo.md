# PyTorch to MARBLE Converter TODO

- [x] Implement converter registry and decorator.
- [x] Create graph builder helpers for neurons and synapses.
  - [x] Fully connected layers
  - [x] Activation handling
- [x] Add default converters for Linear and ReLU layers.
- [x] Build convert_model using torch.fx symbolic tracing.
- [x] Raise explicit errors for unsupported layers.
- [x] Provide CLI script `convert_model.py`.
- [x] Write unit tests covering simple model conversion.
- [x] Add converter for Conv2d layers (single-channel only).
  - [x] Unit test conv2d conversion.
- [x] Document CLI usage in README.

## Upcoming Features

### 0. Core conversion engine
- [ ] Ensure `convert_model` handles functional operations from `torch.nn.functional`
- [x] Graceful fallback when `torch.fx` tracing fails
- [x] Provide clear error when layer type is not registered
- [x] Raise error for unsupported functional operations

### Functional layer converters
- [ ] Registry for functional operations
- [ ] Converter for `F.relu`
- [ ] Converter for `F.sigmoid`
- [ ] Converter for `F.tanh`
- [ ] Unit tests covering functional converters
  - [x] Unsupported functional op error test

### 1. Expand registry with more PyTorch layers
- [x] BatchNorm and Dropout
- [ ] Flatten and Reshape operations
  - [x] Flatten
  - [x] Reshape (Unflatten)
  - [ ] View/reshape functional
- [x] Additional activations (Sigmoid, Tanh, GELU)
- [ ] Multi-channel Conv2d
- [ ] Pooling layers
  - [ ] MaxPool2d and AvgPool2d
  - [ ] GlobalAvgPool2d and adaptive pooling
- [ ] Sequential and ModuleList containers
- [ ] Embedding layers
- [ ] Recurrent layers (RNN, LSTM, GRU)
- [ ] Normalization layers (LayerNorm, GroupNorm)
- [ ] Transformer blocks

### 2. High-level graph construction API
- [ ] Helper to add neuron groups with activations
- [ ] Helper to add synapses with weights and bias
- [ ] Parameterized wrappers for linear and convolutional layers
- [ ] Documentation for graph builder utilities

### 3. Weight and activation handling
- [ ] Extract weights and biases from PyTorch layers
- [ ] Store activation type in neuron metadata
- [ ] Support GPU and CPU weight formats

### 4. Custom layer converter support
- [ ] Decorator-based registration for user-defined layers
- [ ] Example converter for a custom layer
- [ ] Unit tests validating custom converter workflow
- [ ] Document registration mechanism in README

### 5. Dry-run improvements
- [x] Number of neurons and synapses created
- [x] Per-layer mapping information
- [ ] Visualize neuron and synapse counts

### 6. Validation utilities
- [ ] Validate converted models by comparing PyTorch and MARBLE outputs
  - [ ] Unit tests for small networks
  - [ ] Integration test for a custom model
  - [ ] CLI option to run validation automatically

### 7. Additional tooling
- [ ] Support converting `.pt` files directly into `.marble` snapshots
- [ ] Provide auto-inference mode summarizing created graph without saving
- [ ] Command line interface for one-step conversion
- [ ] Programmatic API returning a `Core` object
- [ ] YAML configuration for converter options

### 8. Model loader interface
- [ ] CLI supports `--pytorch`, `--output` and `--dry-run`
- [ ] Programmatic `convert_model` function usable in other scripts
- [ ] Example usage documented in README
- [ ] Load checkpoints saved with `torch.save` automatically

### 9. Graph visualization and inspection
- [ ] Visualize generated MARBLE graph structure
- [ ] Display neuron and synapse counts per layer
- [ ] Interactive tool to inspect neuron parameters

### 10. Error handling and logging
- [x] Unsupported layers raise `"[layer type name] is not supported for conversion"`
- [ ] Logging statements for each conversion step

### 11. Dynamic graph support
- [ ] Map PyTorch control flow to MARBLE dynamic topology
- [ ] Handle evolving neuron and synapse creation during inference
- [ ] Unit tests covering dynamic model conversion paths
