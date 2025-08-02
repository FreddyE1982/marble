# Hierarchical Reinforcement Learning Research

This document summarises algorithms relevant for integrating hierarchical reinforcement learning (HRL) into Marble.

## Options Considered
- **Feudal RL**: Separates managers and workers to achieve subgoal decomposition. Suitable for our global workspace message-passing architecture.
- **Options Framework**: Utilises temporally extended actions. Aligns with our plugin system and can be expressed as high-level policies over Neuronenblitz modules.
- **MAXQ**: Decomposes value functions recursively. Works with discrete tasks and offers theoretical guarantees.

After evaluating these methods we selected the **Options Framework** for initial integration due to its flexibility and compatibility with existing `MarbleQLearningAgent` structures.
An experimental implementation lives in `hierarchical_rl.py` where high-level
options are published through the global workspace and executed by subordinate
policies.
