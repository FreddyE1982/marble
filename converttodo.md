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

### 2. High-level graph construction API
- [ ] Helper to add neuron groups with activations
- [ ] Helper to add synapses with weights and bias
- [ ] Parameterized wrappers for linear and convolutional layers
- [ ] Documentation for graph builder utilities

### 3. Custom layer converter support
- [ ] Decorator-based registration for user-defined layers
- [ ] Example converter for a custom layer
- [ ] Unit tests validating custom converter workflow
- [ ] Document registration mechanism in README

### 4. Dry-run improvements
- [x] Number of neurons and synapses created
- [x] Per-layer mapping information
- [ ] Visualize neuron and synapse counts

### 5. Validation utilities
- [ ] Validate converted models by comparing PyTorch and MARBLE outputs
  - [ ] Unit tests for small networks
  - [ ] Integration test for a custom model
  - [ ] CLI option to run validation automatically

### 6. Additional tooling
- [ ] Support converting `.pt` files directly into `.marble` snapshots
- [ ] Provide auto-inference mode summarizing created graph without saving
