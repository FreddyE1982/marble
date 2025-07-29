# PyTorch to MARBLE Converter TODO

1. Implement converter registry and decorator.
2. Create graph builder helpers for neurons and synapses.
   - Fully connected layers
   - Activation handling
3. Add default converters for Linear and ReLU layers.
4. Build convert_model using torch.fx symbolic tracing.
5. Raise explicit errors for unsupported layers.
6. Provide CLI script `convert_model.py`.
7. Write unit tests covering simple model conversion.
