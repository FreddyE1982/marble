# Context-Aware Attention Mechanisms

This note surveys current attention techniques and presents the design adopted in Marble.

## Existing Mechanisms
- **Additive/Bahdanau Attention**: Computes attention scores with a feed-forward network. Flexible but computationally heavy.
- **Scaled Dot-Product Attention**: Utilised in Transformers; efficient and widely adopted.
- **Self-Attention Variants**: Including multi-head and relative positional encodings.

## Proposed Architecture
Our context-aware attention layer augments standard scaled dot-product attention with a learned context vector. This vector modulates the key and query representations before the dot-product step, enabling focus to shift based on recent self-monitoring signals or episodic context.

The implementation resides in `marble_neuronenblitz.context_attention.ContextAwareAttention`.
The module exposes a minimal API compatible with PyTorch. A trainable context
vector is added to both the query and key inputs before linear projection,
after which a scaled dot-product is computed. Softmax weights are applied to the
value tensor and the computation transparently utilises CUDA when available. The
layer therefore slots into existing models while allowing external signals to
influence attention weights.
The layer exposes `temperature` and `dropout` parameters configurable via
`config.yaml` to control sharpness and regularisation. When
`dynamic_attention_enabled` is set, the temperature adapts based on recent
`message_passing_change` statistics.
