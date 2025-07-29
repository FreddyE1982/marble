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
- [x] Ensure `convert_model` handles functional operations from `torch.nn.functional`
- [x] Document layer mapping registry in developer docs
- [x] Graceful fallback when `torch.fx` tracing fails
- [x] Provide clear error when layer type is not registered
- [x] Raise error for unsupported functional operations

### Functional layer converters
- [x] Registry for functional operations
- [x] Converter for `F.relu`
- [x] Converter for `F.sigmoid`
- [x] Converter for `F.tanh`
- [x] Unit tests covering functional converters
  - [x] Unsupported functional op error test

### 1. Expand registry with more PyTorch layers
- [x] BatchNorm and Dropout
- [x] Flatten and Reshape operations
  - [x] Flatten
  - [x] Reshape (Unflatten)
  - [x] View/reshape functional
- [x] Additional activations (Sigmoid, Tanh, GELU)
- [x] Multi-channel Conv2d
  - [x] Support arbitrary input/output channels
  - [x] Unit test multi-channel conversion
  - [x] Error message for unsupported configuration
- [ ] Pooling layers
  - [x] MaxPool2d and AvgPool2d
  - [x] GlobalAvgPool2d converter
  - [x] Adaptive pooling layers
  - [x] Unit tests for global/adaptive pooling
  - [x] Update docs for pooling support
- [x] Sequential and ModuleList containers
  - [x] Handle torch.nn.Sequential recursion
  - [x] Support ModuleList iteration
  - [x] Unit tests for container handling
- [ ] Embedding layers
  - [x] Basic ``Embedding`` converter
  - [x] ``EmbeddingBag`` support
  - [x] Unit tests for embeddings
  - [ ] Support ``padding_idx`` and ``max_norm`` options
  - [ ] Test embeddings on GPU and CPU
- [x] Recurrent layers (RNN, LSTM, GRU)
  - [x] Converter for ``RNN``
  - [x] Converter for ``LSTM``
  - [x] Converter for ``GRU``
  - [x] Unit tests with tiny sequences
  - [ ] Bidirectional and multi-layer support
  - [ ] Persistent hidden state mapping
- [x] Normalization layers (LayerNorm, GroupNorm)
  - [x] ``LayerNorm`` converter
  - [x] ``GroupNorm`` converter
  - [x] Unit tests for normalization
- [ ] Transformer blocks
  - [ ] Self-attention conversion
  - [ ] Feed-forward sublayers
  - [ ] Positional encoding handling
  - [ ] Integration tests on a small transformer

### 2. High-level graph construction API
- [x] Helper to add neuron groups with activations
- [x] Helper to add synapses with weights and bias
- [ ] Parameterized wrappers for linear and convolutional layers
- [ ] Documentation for graph builder utilities
- [ ] Examples demonstrating dynamic message passing setup

### 3. Weight and activation handling
- [x] Extract weights and biases from PyTorch layers
- [x] Store activation type in neuron metadata
- [ ] Support GPU and CPU weight formats
- [ ] Verify bias neurons are created correctly

### 4. Custom layer converter support
- [x] Decorator-based registration for user-defined layers
- [x] Example converter for a custom layer
- [x] Unit tests validating custom converter workflow
- [x] Document registration mechanism in README

### 5. Dry-run improvements
- [x] Number of neurons and synapses created
- [x] Per-layer mapping information
- [ ] Visualize neuron and synapse counts

### 6. Validation utilities
- [ ] Validate converted models by comparing PyTorch and MARBLE outputs
  - [ ] Unit tests for small networks
  - [ ] Integration test for a custom model
  - [ ] CLI option to run validation automatically
- [ ] Numerical tolerances for output comparison

### 7. Additional tooling
- [x] Support converting `.pt` files directly into `.marble` snapshots
- [ ] Provide auto-inference mode summarizing created graph without saving
- [ ] Command line interface for one-step conversion
- [ ] Programmatic API returning a `Core` object
- [ ] YAML configuration for converter options

### 8. Model loader interface
- [x] CLI supports `--pytorch`, `--output` and `--dry-run`
- [x] Programmatic `convert_model` function usable in other scripts
- [x] Example usage documented in README
- [x] Load checkpoints saved with `torch.save` automatically
- [x] Support `.json` or `.marble` output based on extension

### 9. Graph visualization and inspection
- [ ] Visualize generated MARBLE graph structure
- [ ] Display neuron and synapse counts per layer
- [ ] Interactive tool to inspect neuron parameters

### 10. Error handling and logging
- [x] Unsupported layers raise `"[layer type name] is not supported for conversion"`
- [x] Logging statements for each conversion step
  - [x] Consistent error message for invalid Conv2d parameters

### 11. Dynamic graph support
- [ ] Map PyTorch control flow to MARBLE dynamic topology
- [ ] Handle evolving neuron and synapse creation during inference
- [ ] Unit tests covering dynamic model conversion paths
- [ ] Research torch.fx support for control flow constructs

### 12. Universal converter roadmap
- [ ] Comprehensive layer mapping registry
  - [ ] Enumerate all built-in nn layers and map to converters
  - [ ] Provide template for unsupported layers to raise errors
- [ ] Graph construction utilities bridging to dynamic message passing
  - [x] Helper to spawn neurons for input/output dimensions
  - [x] Helper to connect neurons with weighted synapses
  - [ ] Activation flag storage for message passing
- [ ] torch.fx integration for arbitrary models
  - [ ] Trace custom layers and call registered converters
  - [ ] Allow decorators to register new converters
- [ ] Weight and bias extraction helpers
  - [ ] Handle GPU tensors transparently
  - [ ] Inject bias neurons with correct values
- [ ] Conversion CLI and API enhancements
  - [ ] Option to produce .marble snapshot directly
  - [ ] Auto-inference mode printing neuron/synapse counts
  - [ ] Validation mode comparing outputs with PyTorch
- [ ] Validation suite comparing PyTorch and MARBLE outputs
  - [ ] Per-layer unit tests
  - [ ] End-to-end comparison for simple models
