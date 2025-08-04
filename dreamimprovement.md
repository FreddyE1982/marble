# Dream Improvement Implementation Plan

This document outlines the detailed steps required to transform Marble's current "dreaming" mechanism into a constructive memory consolidation process. Each large task is decomposed into manageable subtasks to guide development.

## 1. Establish Replay and Memory Tracking Infrastructure
1.1 **Create Dream Replay Buffer**
  - 1.1.1 Define a data structure to store past experiences (input, target, reward, emotion, timestamp).
  - 1.1.2 Implement FIFO capacity and importance-based eviction (e.g., oldest or lowest-salience).
  - 1.1.3 Expose buffer controls via configuration YAML and update `yaml-manual.txt` and `CONFIGURABLE_PARAMETERS.md`.

1.2 **Tag Experiences With Emotion/Arousal**
  - 1.2.1 Extend episode objects to include neuromodulatory tags (arousal, stress, reward, emotion).
  - 1.2.2 Update all data ingestion and training pathways to populate these tags.
  - 1.2.3 Add validation to ensure tags fall within allowed ranges.

## 2. Implement Dream Cycle Operations
2.1 **Memory Replay / Consolidation**
  - 2.1.1 During a dream cycle, sample batches from the replay buffer.
  - 2.1.2 Run forward and backward passes to reinforce correct associations using gradient descent.
  - 2.1.3 Log training metrics (loss, accuracy) for dream cycles.

2.2 **Emotion and Salience Weighting**
  - 2.2.1 Compute sampling probabilities biased toward higher emotion/arousal scores.
  - 2.2.2 Allow configurable weighting functions (linear, exponential).
  - 2.2.3 Expose weighting parameters in YAML and update manuals and defaults.

2.3 **Mental Housekeeping / Pruning**
  - 2.3.1 Identify low-importance connections using weight magnitude or contribution statistics.
  - 2.3.2 Apply mild decay or L1/L2 regularization to these connections during dreams.
  - 2.3.3 Optionally inject noise or recombine fragments from multiple memories to encourage novelty.

2.4 **Instant Replay of Recent Events**
  - 2.4.1 Maintain a short-term FIFO buffer focused on most recent episodes.
  - 2.4.2 Prioritize sampling from this buffer early in each dream cycle.
  - 2.4.3 Merge results into long-term replay buffer after processing.

## 3. Integrate With Runtime and Configuration
3.1 **Dream Scheduler**
  - 3.1.1 Modify `start_dreaming` to orchestrate replay, weighting, housekeeping, and instant replay steps sequentially.
  - 3.1.2 Allow configurable interval and batch sizes for dream cycles.
  - 3.1.3 Ensure GPU/CPU compatibility for all new operations.

3.2 **Neuromodulatory Interaction**
  - 3.2.1 Derive decay/regularization rates from neuromodulatory system values (arousal, stress).
  - 3.2.2 Allow dreams to adjust neuromodulatory state based on replay outcomes.

3.3 **Snapshot Persistence**
  - 3.3.1 Verify that replay buffers and neuromodulatory states serialize correctly when saving snapshots.
  - 3.3.2 Update load routines to restore dreaming state.

## 4. Testing and Validation
4.1 **Unit Tests**
  - 4.1.1 Add tests for replay buffer behavior (insert, evict, sample by salience).
  - 4.1.2 Test dream cycle operations: memory consolidation improves accuracy on replayed data.
  - 4.1.3 Validate housekeeping decreases unused weight magnitudes without harming performance.

4.2 **Integration Tests**
  - 4.2.1 Simulate training + dreaming workflow and verify persistence after snapshot save/load.
  - 4.2.2 Ensure GPU and CPU paths produce equivalent results within tolerance.

4.3 **Benchmarking**
  - 4.3.1 Measure model performance with and without dreaming to confirm improvements.
  - 4.3.2 Log metrics for memory retention and learning stability over time.

## 5. Documentation and Tutorials
5.1 Update `yaml-manual.txt`, `CONFIGURABLE_PARAMETERS.md`, and `TUTORIAL.md` to cover new dreaming options with detailed explanations and examples.

5.2 Provide a step-by-step tutorial project demonstrating improved dreaming using a real dataset and code snippets for downloading, training, and observing dream effects.

5.3 Record any remaining tasks in `TODO.md` if implementation exceeds current scope.

---
Following this roadmap will transform Marble's dreaming from weight decay into a biologically inspired consolidation process that strengthens memories while cleaning up unused connections.
