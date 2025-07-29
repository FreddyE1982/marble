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
- [ ] Expand registry with more PyTorch layers
  - [x] BatchNorm and Dropout
  - [ ] Flatten and Reshape operations
    - [x] Flatten
    - [ ] Reshape
  - [x] Additional activations (Sigmoid, Tanh, GELU)
  - [ ] Multi-channel Conv2d
  - [ ] MaxPool2d and AvgPool2d
  - [ ] GlobalAvgPool2d and Adaptive pooling layers
  - [ ] Sequential and ModuleList containers
- [ ] Create high-level graph construction API
  - [ ] Helper to add neuron groups with activations
  - [ ] Helper to add synapses with weights and bias
  - [ ] Parameterized wrappers for linear and convolutional layers
- [ ] Support custom layer converters via decorator registration
  - [ ] Example converter for a user-defined PyTorch layer
  - [ ] Unit tests for custom converter workflow
- [ ] Enhance `dry_run` mode to output summary statistics
  - [ ] Number of neurons and synapses created
  - [ ] Per-layer mapping information
- [ ] Validate converted models by comparing PyTorch and MARBLE outputs
  - [ ] Unit tests for small networks
  - [ ] Integration test for custom model
