# Neuronenblitz Package Overview

This package contains the implementation of the adaptive learning system used by MARBLE. The modules are organised as follows:

- `core.py` – main `Neuronenblitz` class with dynamic wandering and synaptic plasticity logic.
- `learning.py` – reinforcement learning helpers used by `Neuronenblitz` and the high level agents.
- `memory.py` – utilities for managing memory gate strengths, episodic path replay, and eligibility traces.
- `__init__.py` – exports commonly used functions and the core class.

These modules can be imported individually, allowing advanced users to extend or replace specific functionality.

Additional plugin modules implement context-aware attention, theory-of-mind reasoning and predictive coding. Activate them with the respective `activate` functions to augment the core algorithms.
