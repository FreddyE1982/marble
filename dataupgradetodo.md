# Data Loader and Tokenizer Upgrade TODO

This document outlines the implementation plan for reworking data loading, training, and inference as described in the user request. Each step is broken down into sub‑steps so that no single sub‑step is too large for one agent run.

## 1. Tokenizer‑less Mode
1.1 Extend `DataLoader` to track metadata for each encoded object so decoding restores the exact Python type and format without loss.
1.2 Add round‑trip checks during training to verify that decoded outputs match the originals.
&nbsp;&nbsp;1.2.1 If the decoded value does not equal the original value, apply a configurable loss penalty.
1.3 Update training loops (e.g., `Brain.train` and learners) to use the new round‑trip check.
1.4 Modify inference paths so that `Brain.infer` returns objects decoded via `DataLoader.decode` automatically.
1.5 Write comprehensive unit tests for encoding/decoding of various object pairs (text, images, audio, binary blobs, etc.) and for the round‑trip penalty mechanism.

## 2. Hugging Face Tokenizer Mode
2.1 Integrate the `tokenizers` library as an optional dependency.
&nbsp;&nbsp;2.1.1 Update `requirements.txt` and add installation instructions.
2.2 Create a new tokenizer utility module to load or build tokenizers using the HF pipeline (normalization, pre‑tokenization, model, post‑processing).
&nbsp;&nbsp;2.2.1 Support loading tokenizers from JSON files as in the HF quick tour.
&nbsp;&nbsp;2.2.2 Provide convenience functions to train WordPiece/BPE/Unigram models.
&nbsp;&nbsp;2.2.3 Include support for the built‑in tokenizers from the `tokenizers` library (`BertWordPieceTokenizer`, `ByteLevelBPETokenizer`, `CharBPETokenizer`, `SentencePieceBPETokenizer`, `SentencePieceUnigramTokenizer`).
&nbsp;&nbsp;&nbsp;&nbsp;2.2.3.1 Allow selecting these implementations via YAML and programmatic interfaces.
2.3 Extend `DataLoader` with an optional tokenizer object.
&nbsp;&nbsp;2.3.1 When a tokenizer is provided, `encode` converts string inputs to token IDs and stores them as tensors.
&nbsp;&nbsp;2.3.2 `decode` reverses the process using the same tokenizer.
&nbsp;&nbsp;2.3.3 Provide helper functions to instantiate built‑in tokenizers by name so configuration files can reference them directly.
2.4 Ensure that any tokenizer used for training is embedded directly into Marble model checkpoints.
&nbsp;&nbsp;2.4.1 Modify save/load routines to serialize the tokenizer JSON along with model parameters.
&nbsp;&nbsp;2.4.2 On load, reconstruct the tokenizer from the embedded data without requiring external files.
2.5 Update `infer` helpers so that `infer(input)` decodes outputs using the embedded tokenizer, while `infer(input, tensor=True)` bypasses decoding and returns the raw tensor.
2.6 Add unit tests covering tokenizer training, embedding into checkpoints, and both inference modes.
&nbsp;&nbsp;2.6.1 Include tests for each supported built‑in tokenizer to ensure they can be serialized and reconstructed correctly.

## 3. Configuration and Documentation
3.1 Add YAML configuration options to select between tokenizer‑less and HF tokenizer modes and to specify tokenizer parameters (model type, vocabulary size, etc.).
3.2 Document new parameters in `CONFIGURABLE_PARAMETERS.md` and expand `yaml-manual.txt` with detailed explanations.
3.3 Provide tutorial updates in `TUTORIAL.md` showing how to use both modes with real datasets.

## 4. Migration and Compatibility
4.1 Update existing dataset loading functions (`load_dataset`, `load_hf_dataset`) to interface with the enhanced `DataLoader`.
4.2 Ensure backward compatibility with previously saved Marble models by supporting missing tokenizer data (assume tokenizer‑less mode when absent).
4.3 Write migration tests to load old checkpoints and verify they still decode correctly.

