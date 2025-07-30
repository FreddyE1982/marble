# Data Loader and Tokenizer Upgrade TODO

This document outlines the implementation plan for reworking data loading, training, and inference as described in the user request. Each step is broken down into sub‑steps so that no single sub‑step is too large for one agent run.

## 1. Tokenizer‑less Mode
- [x] Extend `DataLoader` to track metadata for each encoded object so decoding restores the exact Python type and format without loss.
- [x] Add round‑trip checks during training to verify that decoded outputs match the originals.
  - [x] If the decoded value does not equal the original value, apply a configurable loss penalty.
- [x] Update training loops (e.g., `Brain.train` and learners) to use the new round‑trip check.
- [x] Modify inference paths so that `Brain.infer` returns objects decoded via `DataLoader.decode` automatically.
- [x] Write comprehensive unit tests for encoding/decoding of various object pairs (text, images, audio, binary blobs, etc.) and for the round‑trip penalty mechanism.

## 2. Hugging Face Tokenizer Mode
- [x] Integrate the `tokenizers` library as an optional dependency.
  - [x] Update `requirements.txt` and add installation instructions.
- [x] Create a new tokenizer utility module to load or build tokenizers using the HF pipeline (normalization, pre‑tokenization, model, post‑processing).
  - [x] Support loading tokenizers from JSON files as in the HF quick tour.
  - [x] Provide convenience functions to train WordPiece/BPE/Unigram models.
  - [x] Include support for the built‑in tokenizers from the `tokenizers` library (`BertWordPieceTokenizer`, `ByteLevelBPETokenizer`, `CharBPETokenizer`, `SentencePieceBPETokenizer`, `SentencePieceUnigramTokenizer`).
    - [x] Allow selecting these implementations via YAML and programmatic interfaces.
- [x] Extend `DataLoader` with an optional tokenizer object.
  - [x] When a tokenizer is provided, `encode` converts string inputs to token IDs and stores them as tensors.
  - [x] `decode` reverses the process using the same tokenizer.
  - [x] Provide helper functions to instantiate built‑in tokenizers by name so configuration files can reference them directly.
- [x] Ensure that any tokenizer used for training is embedded directly into Marble model checkpoints.
  - [x] Modify save/load routines to serialize the tokenizer JSON along with model parameters.
  - [x] On load, reconstruct the tokenizer from the embedded data without requiring external files.
- [x] Update `infer` helpers so that `infer(input)` decodes outputs using the embedded tokenizer, while `infer(input, tensor=True)` bypasses decoding and returns the raw tensor.
- [x] Add unit tests covering tokenizer training, embedding into checkpoints, and both inference modes.
  - [x] Include tests for each supported built‑in tokenizer to ensure they can be serialized and reconstructed correctly.

## 3. Configuration and Documentation
- [x] Add YAML configuration options to select between tokenizer‑less and HF tokenizer modes and to specify tokenizer parameters (model type, vocabulary size, etc.).
- [x] Document new parameters in `CONFIGURABLE_PARAMETERS.md` and expand `yaml-manual.txt` with detailed explanations.
- [x] Provide tutorial updates in `TUTORIAL.md` showing how to use both modes with real datasets.

## 4. Migration and Compatibility
- [x] Update existing dataset loading functions (`load_dataset`, `load_hf_dataset`) to interface with the enhanced `DataLoader`.
- [x] Ensure backward compatibility with previously saved Marble models by supporting missing tokenizer data (assume tokenizer‑less mode when absent).
- [x] Write migration tests to load old checkpoints and verify they still decode correctly.

