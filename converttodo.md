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
        - [ ] Align parameter ordering between PyTorch and MARBLE.
        - [ ] Validate hidden state concatenation.
      - [ ] Handle stacked layers with appropriate parameter naming.
        - [ ] Prefix layer indices consistently.
        - [ ] Ensure loading preserves original hierarchy.
      - [ ] Add tests converting multi-layer bidirectional models.
        - [ ] Build example network with two bidirectional layers.
        - [ ] Compare outputs against PyTorch reference.
    - [ ] Persistent hidden state mapping
      - [ ] Serialize initial hidden states with layer metadata.
        - [ ] Embed state tensors into converter output.
        - [ ] Record correspondence to layer identifiers.
      - [ ] Restore hidden states during MARBLE execution.
        - [ ] Load serialized states into runtime structures.
        - [ ] Verify shapes match original expectations.
      - [ ] Provide tests verifying state persistence across runs.
        - [ ] Save model, reload, and compare hidden states.
        - [ ] Ensure no drift after multiple executions.
- [x] Normalization layers (LayerNorm, GroupNorm)
  - [x] ``LayerNorm`` converter
  - [x] ``GroupNorm`` converter
  - [x] Unit tests for normalization
   - [ ] Transformer blocks
    - [ ] Self-attention conversion
      - [ ] Translate query, key, and value projections.
        - [ ] Map weight matrices and biases separately.
        - [ ] Ensure projection shapes align with head count.
      - [ ] Support multi-head attention weight splitting.
        - [ ] Split combined matrices into per-head chunks.
        - [ ] Recombine outputs after attention.
      - [ ] Add unit tests for attention weight parity.
        - [ ] Verify numerical parity for single-head case.
        - [ ] Extend tests to multi-head scenarios.
    - [ ] Feed-forward sublayers
      - [ ] Convert linear1 and linear2 layers with activation.
        - [ ] Preserve activation type and placement.
        - [ ] Translate weight and bias tensors.
      - [ ] Handle dropout placement within sublayer.
        - [ ] Detect dropout modules during tracing.
        - [ ] Insert dropout nodes into MARBLE graph.
      - [ ] Verify output numerically against PyTorch.
        - [ ] Run random inputs through both models.
        - [ ] Check differences within tolerance.
    - [ ] Positional encoding handling
      - [ ] Map sinusoidal and learned positional embeddings.
        - [ ] Convert embedding tensors and scaling factors.
        - [ ] Support optional padding tokens.
      - [ ] Expose positional encoding choice via config.
        - [ ] Add config parameter and CLI option.
        - [ ] Document when to use each encoding type.
      - [ ] Add tests ensuring positions align after conversion.
        - [ ] Compare embedding indices between models.
        - [ ] Validate sequence lengths remain unchanged.
    - [ ] Integration tests on a small transformer
      - [ ] Convert a tiny transformer model end-to-end.
        - [ ] Trace model and run through converter.
        - [ ] Load converted model into MARBLE runtime.
      - [ ] Compare MARBLE and PyTorch outputs for sample inputs.
        - [ ] Generate deterministic test inputs.
        - [ ] Assert outputs differ within tolerance.
      - [ ] Document transformer conversion workflow.
        - [ ] Provide step-by-step guide in README.
        - [ ] Include troubleshooting section.

### 2. High-level graph construction API
- [x] Helper to add neuron groups with activations
- [x] Helper to add synapses with weights and bias
- [x] Parameterized wrappers for linear and convolutional layers
  - [x] ``linear_layer`` wrapper
  - [x] ``conv2d_layer`` wrapper
- [x] Documentation for graph builder utilities
   - [ ] Examples demonstrating dynamic message passing setup
    - [ ] Create example script building a dynamic message passing graph.
        - [ ] Showcase runtime addition and removal of edges.
        - [ ] Include comments explaining each step.
    - [ ] Document example usage in README.
        - [ ] Provide command to run the example.
        - [ ] Explain expected output and graph behavior.
    - [ ] Add unit test covering dynamic example conversion.
        - [ ] Verify script executes without errors.
        - [ ] Assert generated graph matches expected structure.

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
        - [ ] Compare single-layer models across frameworks.
        - [ ] Check parameter copying correctness.
      - [ ] Integration test for a custom model
        - [ ] Convert a moderate-size model end-to-end.
        - [ ] Validate numerical equivalence on sample data.
      - [ ] CLI option to run validation automatically
          - [ ] Add `--validate` flag to `convert_model.py`.
          - [ ] Execute PyTorch vs MARBLE comparison when the flag is provided.
          - [ ] Output a validation report highlighting layer mismatches.
          - [ ] Document the validation flag in README and CLI help.
  - [ ] Numerical tolerances for output comparison
      - [ ] Define default tolerance thresholds for floating point comparisons.
        - [ ] Determine separate tolerances for CPU and GPU.
        - [ ] Justify chosen defaults in docs.
      - [ ] Allow overriding tolerance via CLI and configuration.
        - [ ] Add `--tolerance` flag and config entry.
        - [ ] Validate user-provided values.
      - [ ] Document tolerance settings in README.
        - [ ] Include examples illustrating effects of tolerance.
        - [ ] Warn about overly strict values causing failures.
      - [ ] Add tests covering tolerance edge cases.
        - [ ] Test near-threshold differences.
        - [ ] Confirm override works via CLI and config.

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
- [x] Visualize generated MARBLE graph structure
  - [x] Integrate graph visualization library (e.g., pyvis or plotly).
  - [x] Save rendered graphs to HTML for inspection.
  - [x] Add tests verifying graph export.
- [x] Display neuron and synapse counts per layer
  - [x] Implement summarizer that tallies counts per layer.
  - [x] Provide CLI option to print or save counts.
  - [x] Write unit tests for summarizer accuracy.
  - [ ] Interactive tool to inspect neuron parameters
    - [ ] Build Streamlit viewer with filtering and search.
        - [ ] Implement sidebar controls for filtering by layer or type.
        - [ ] Provide search box for neuron IDs.
    - [ ] Support selecting neurons to show detailed parameters.
        - [ ] Display weights, biases, and activation info.
        - [ ] Allow exporting selected neuron data.
    - [ ] Add GUI tests for viewer components.
        - [ ] Simulate user interaction with filters and search.
        - [ ] Ensure detail view renders without errors.

### 10. Error handling and logging
- [x] Unsupported layers raise `"[layer type name] is not supported for conversion"`
- [x] Logging statements for each conversion step
  - [x] Consistent error message for invalid Conv2d parameters

### 11. Dynamic graph support
  - [ ] Map PyTorch control flow to MARBLE dynamic topology
    - [ ] Parse torch.fx conditional and loop nodes.
        - [ ] Identify node types representing branches and loops.
        - [ ] Convert nodes into intermediate representation.
    - [ ] Generate dynamic neuron groups for branches.
        - [ ] Create separate groups for each branch path.
        - [ ] Merge outputs using switch-like connections.
    - [ ] Document limitations and edge cases.
        - [ ] Note unsupported control-flow patterns.
        - [ ] Provide workarounds where possible.
  - [ ] Handle evolving neuron and synapse creation during inference
    - [ ] Implement runtime graph mutation utilities.
        - [ ] Support adding and removing nodes on the fly.
        - [ ] Maintain consistency of references and indices.
    - [ ] Ensure thread safety during dynamic updates.
        - [ ] Guard mutations with locks or queues.
        - [ ] Add checks for concurrent modification.
    - [ ] Add tests exercising dynamic growth.
        - [ ] Simulate incremental graph expansion during inference.
            - [ ] Build test model that adds nodes at runtime.
            - [ ] Trigger growth while evaluating sample inputs.
        - [ ] Verify outputs remain stable.
            - [ ] Compare predictions before and after growth.
            - [ ] Assert no divergence beyond tolerance.
  - [ ] Unit tests covering dynamic model conversion paths
    - [ ] Create mock models with branching control flow.
        - [ ] Include nested loops and conditionals.
    - [ ] Verify converted graphs execute correctly.
        - [ ] Run forward passes and compare outputs.
  - [ ] Research torch.fx support for control flow constructs
    - [ ] Survey existing torch.fx features for loops and conditionals.
        - [ ] Review documentation and open issues.
    - [ ] Prototype tracing of a model using control flow.
        - [ ] Evaluate limitations encountered.
    - [ ] Summarize findings in documentation.
        - [ ] Provide recommendations for future work.

### 12. Universal converter roadmap
- [ ] Comprehensive layer mapping registry
  - [ ] Enumerate all built-in nn layers and map to converters
      - [ ] List available layers from PyTorch documentation.
      - [ ] Match existing converters to these layers.
      - [ ] Flag layers missing converter implementations.
  - [ ] Provide template for unsupported layers to raise errors
      - [ ] Define standard error message format.
      - [ ] Create helper generating error stubs for new layers.
      - [ ] Document template usage for contributors.
- [ ] Graph construction utilities bridging to dynamic message passing
  - [x] Helper to spawn neurons for input/output dimensions
  - [x] Helper to connect neurons with weighted synapses
    - [x] Activation flag storage for message passing
      - [x] Add boolean flag field to neuron metadata.
      - [x] Propagate activation flags through graph builder.
      - [x] Document usage for runtime evaluators.
      - [x] Add tests verifying flag presence.
- [ ] torch.fx integration for arbitrary models
  - [ ] Trace custom layers and call registered converters
      - [ ] Extend tracing logic to capture user-defined layers.
      - [ ] Look up converters via registry and invoke them.
      - [ ] Raise descriptive errors when converters are missing.
  - [ ] Allow decorators to register new converters
      - [ ] Design decorator API for converter registration.
      - [ ] Register decorated functions at import time.
      - [ ] Provide examples demonstrating decorator usage.
- [ ] Weight and bias extraction helpers
  - [ ] Handle GPU tensors transparently
      - [ ] Detect tensor device during extraction.
      - [ ] Move GPU tensors to CPU when necessary.
      - [ ] Preserve original device information for reconstruction.
  - [ ] Inject bias neurons with correct values
      - [ ] Create bias neuron nodes within graph builder.
      - [ ] Populate bias nodes with corresponding weights.
      - [ ] Validate bias application through unit tests.
- [ ] Conversion CLI and API enhancements
  - [ ] Option to produce .marble snapshot directly
      - [ ] Add CLI flag enabling snapshot output.
      - [ ] Serialise converted graph into `.marble` format.
      - [ ] Ensure snapshot includes required metadata.
  - [ ] Auto-inference mode printing neuron/synapse counts
      - [ ] Implement flag triggering automatic inference.
      - [ ] Compute neuron and synapse counts after conversion.
      - [ ] Display counts in CLI output.
  - [ ] Validation mode comparing outputs with PyTorch
      - [ ] Run forward pass on original PyTorch model.
      - [ ] Run same inputs through converted MARBLE model.
      - [ ] Report differences and fail on large deviations.
- [ ] Validation suite comparing PyTorch and MARBLE outputs
  - [ ] Per-layer unit tests
      - [ ] Select representative layers for testing.
      - [ ] Verify converted outputs match PyTorch results.
  - [ ] End-to-end comparison for simple models
      - [ ] Convert small reference model to MARBLE.
      - [ ] Execute model in both frameworks on sample data.
      - [ ] Compare predictions and document parity.

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
    - [ ] Set up dedicated Streamlit page with CPU/GPU selection.
    - [ ] Load converted weight tensors and render them as heatmaps.
    - [ ] Display layer metadata alongside visualisations.
  - [ ] Provide search and filtering capabilities.
    - [ ] Add text search to locate layers by name.
    - [ ] Include sliders to filter by weight magnitude or index range.
  - [ ] Write GUI tests ensuring viewer loads converted weights.
    - [ ] Test loading of weights from a sample converted model.
    - [ ] Validate search and filter controls update the view.
