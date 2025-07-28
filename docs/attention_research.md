# Context-Aware Attention Mechanisms

This note surveys current attention techniques and presents the design adopted in Marble.

## Existing Mechanisms
- **Additive/Bahdanau Attention**: Computes attention scores with a feed-forward network. Flexible but computationally heavy.
- **Scaled Dot-Product Attention**: Utilised in Transformers; efficient and widely adopted.
- **Self-Attention Variants**: Including multi-head and relative positional encodings.

## Proposed Architecture
Our context-aware attention layer augments standard scaled dot-product attention with a learned context vector. This vector modulates the key and query representations before the dot-product step, enabling focus to shift based on recent self-monitoring signals or episodic context.

The implementation resides in `marble_neuronenblitz.context_attention.ContextAwareAttention`.
