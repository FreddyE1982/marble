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
- [x] Pooling layers
  - [x] MaxPool2d and AvgPool2d
  - [x] GlobalAvgPool2d converter
  - [x] Adaptive pooling layers
  - [x] Unit tests for global/adaptive pooling
  - [x] Update docs for pooling support
- [x] Sequential and ModuleList containers
  - [x] Handle torch.nn.Sequential recursion
  - [x] Support ModuleList iteration
  - [x] Unit tests for container handling
- [x] Embedding layers
  - [x] Basic ``Embedding`` converter
  - [x] ``EmbeddingBag`` support
  - [x] Unit tests for embeddings
  - [x] Support ``padding_idx`` and ``max_norm`` options
  - [x] Test embeddings on GPU and CPU
- [x] Recurrent layers (RNN, LSTM, GRU)
  - [x] Converter for ``RNN``
  - [x] Converter for ``LSTM``
  - [x] Converter for ``GRU``
  - [x] Unit tests with tiny sequences
  - [ ] Bidirectional and multi-layer support
    - [ ] Map forward and backward weights for bidirectional RNNs.
    - [ ] Handle stacked layers with appropriate parameter naming.
    - [ ] Add tests converting multi-layer bidirectional models.
  - [ ] Persistent hidden state mapping
    - [ ] Serialize initial hidden states with layer metadata.
    - [ ] Restore hidden states during MARBLE execution.
    - [ ] Provide tests verifying state persistence across runs.
- [x] Normalization layers (LayerNorm, GroupNorm)
  - [x] ``LayerNorm`` converter
  - [x] ``GroupNorm`` converter
  - [x] Unit tests for normalization
 - [ ] Transformer blocks
  - [ ] Self-attention conversion
    - [ ] Translate query, key, and value projections.
    - [ ] Support multi-head attention weight splitting.
    - [ ] Add unit tests for attention weight parity.
  - [ ] Feed-forward sublayers
    - [ ] Convert linear1 and linear2 layers with activation.
    - [ ] Handle dropout placement within sublayer.
    - [ ] Verify output numerically against PyTorch.
  - [ ] Positional encoding handling
    - [ ] Map sinusoidal and learned positional embeddings.
    - [ ] Expose positional encoding choice via config.
    - [ ] Add tests ensuring positions align after conversion.
  - [ ] Integration tests on a small transformer
    - [ ] Convert a tiny transformer model end-to-end.
    - [ ] Compare MARBLE and PyTorch outputs for sample inputs.
    - [ ] Document transformer conversion workflow.

### 2. High-level graph construction API
- [x] Helper to add neuron groups with activations
- [x] Helper to add synapses with weights and bias
- [x] Parameterized wrappers for linear and convolutional layers
  - [x] ``linear_layer`` wrapper
  - [x] ``conv2d_layer`` wrapper
- [x] Documentation for graph builder utilities
 - [ ] Examples demonstrating dynamic message passing setup
  - [ ] Create example script building a dynamic message passing graph.
  - [ ] Document example usage in README.
  - [ ] Add unit test covering dynamic example conversion.

### 3. Weight and activation handling
- [x] Extract weights and biases from PyTorch layers
- [x] Store activation type in neuron metadata
 - [x] Support GPU and CPU weight formats
 - [x] Verify bias neurons are created correctly

### 4. Custom layer converter support
- [x] Decorator-based registration for user-defined layers
- [x] Example converter for a custom layer
- [x] Unit tests validating custom converter workflow
- [x] Document registration mechanism in README

### 5. Dry-run improvements
- [x] Number of neurons and synapses created
- [x] Per-layer mapping information
- [x] Visualize neuron and synapse counts

- [x] Add `--summary` CLI flag to print dry-run stats
- [x] Support saving summary to JSON via `--summary-output`
### 6. Validation utilities
- [ ] Validate converted models by comparing PyTorch and MARBLE outputs
    - [ ] Unit tests for small networks
    - [ ] Integration test for a custom model
    - [ ] CLI option to run validation automatically
        - [ ] Add `--validate` flag to `convert_model.py`.
        - [ ] Execute PyTorch vs MARBLE comparison when the flag is provided.
        - [ ] Output a validation report highlighting layer mismatches.
        - [ ] Document the validation flag in README and CLI help.
- [ ] Numerical tolerances for output comparison
    - [ ] Define default tolerance thresholds for floating point comparisons.
    - [ ] Allow overriding tolerance via CLI and configuration.
    - [ ] Document tolerance settings in README.
    - [ ] Add tests covering tolerance edge cases.

### 7. Additional tooling
- [x] Support converting `.pt` files directly into `.marble` snapshots
- [x] Provide auto-inference mode summarizing created graph without saving
- [x] Command line interface for one-step conversion
- [x] Programmatic API returning a `Core` object
- [x] YAML configuration for converter options

### 8. Model loader interface
- [x] CLI supports `--pytorch`, `--output` and `--dry-run`
- [x] Programmatic `convert_model` function usable in other scripts
- [x] Example usage documented in README
- [x] Load checkpoints saved with `torch.save` automatically
- [x] Support `.json` or `.marble` output based on extension

### 9. Graph visualization and inspection
- [ ] Visualize generated MARBLE graph structure
  - [ ] Integrate graph visualization library (e.g., pyvis or plotly).
  - [ ] Save rendered graphs to HTML for inspection.
  - [ ] Add tests verifying graph export.
- [x] Display neuron and synapse counts per layer
  - [x] Implement summarizer that tallies counts per layer.
  - [x] Provide CLI option to print or save counts.
  - [x] Write unit tests for summarizer accuracy.
- [ ] Interactive tool to inspect neuron parameters
  - [ ] Build Streamlit viewer with filtering and search.
  - [ ] Support selecting neurons to show detailed parameters.
  - [ ] Add GUI tests for viewer components.

### 10. Error handling and logging
- [x] Unsupported layers raise `"[layer type name] is not supported for conversion"`
- [x] Logging statements for each conversion step
  - [x] Consistent error message for invalid Conv2d parameters

### 11. Dynamic graph support
- [ ] Map PyTorch control flow to MARBLE dynamic topology
  - [ ] Parse torch.fx conditional and loop nodes.
  - [ ] Generate dynamic neuron groups for branches.
  - [ ] Document limitations and edge cases.
- [ ] Handle evolving neuron and synapse creation during inference
  - [ ] Implement runtime graph mutation utilities.
  - [ ] Ensure thread safety during dynamic updates.
  - [ ] Add tests exercising dynamic growth.
- [ ] Unit tests covering dynamic model conversion paths
  - [ ] Create mock models with branching control flow.
  - [ ] Verify converted graphs execute correctly.
- [ ] Research torch.fx support for control flow constructs
  - [ ] Survey existing torch.fx features for loops and conditionals.
  - [ ] Prototype tracing of a model using control flow.
  - [ ] Summarize findings in documentation.

### 12. Universal converter roadmap
- [ ] Comprehensive layer mapping registry
  - [ ] Enumerate all built-in nn layers and map to converters
  - [ ] Provide template for unsupported layers to raise errors
- [ ] Graph construction utilities bridging to dynamic message passing
  - [x] Helper to spawn neurons for input/output dimensions
  - [x] Helper to connect neurons with weighted synapses
    - [ ] Activation flag storage for message passing
      - [ ] Add boolean flag field to neuron metadata.
      - [ ] Propagate activation flags through graph builder.
      - [ ] Document usage for runtime evaluators.
      - [ ] Add tests verifying flag presence.
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

### 13. Dynamic message passing integration
- [ ] Store activation behavior metadata in neurons for runtime evaluation
  - [ ] Extend neuron data structures with activation descriptors.
  - [ ] Ensure converters populate metadata.
  - [ ] Add tests validating metadata availability.
- [ ] Plugin or extension to handle activations during message passing
  - [ ] Define plugin API for custom activation behaviors.
  - [ ] Provide default plugin implementing common activations.
  - [ ] Include tests for plugin registration.
- [ ] Example script demonstrating dynamic message passing with converted networks
  - [ ] Build sample network utilizing activation plugin.
  - [ ] Show runtime evaluation using metadata.
  - [ ] Document script in examples directory.

### 14. Control flow and topology adaptation
- [ ] Convert if/else branches to dynamic neuron creation
  - [ ] Detect branch conditions during tracing.
  - [ ] Generate neurons for each branch path.
  - [ ] Merge results with switch-like synapses.
- [ ] Support loops by reusing neuron groups per iteration
  - [ ] Identify loop bodies in torch.fx graphs.
  - [ ] Implement mechanism to reuse neuron structures.
  - [ ] Handle loop termination criteria.
- [ ] Unit test a model with simple control flow
  - [ ] Build toy model containing branch and loop.
  - [ ] Verify converted dynamic graph produces same output.
  - [ ] Document test scenario.

### 15. Visualization enhancements
- [ ] Provide CLI flag to visualize generated MARBLE graph
  - [ ] Add `--show-graph` option to converter CLI.
  - [ ] Render graph after conversion when flag is set.
  - [ ] Include tests verifying flag triggers visualization.
- [x] Save neuron and synapse counts per layer to .csv
  - [x] Implement exporter writing counts to CSV.
  - [x] Add CLI argument for output path.
  - [x] Test CSV output formatting.
- [ ] Interactive viewer to inspect weights
  - [ ] Develop Streamlit app to browse layer weights.
  - [ ] Provide search and filtering capabilities.
  - [ ] Write GUI tests ensuring viewer loads converted weights.
