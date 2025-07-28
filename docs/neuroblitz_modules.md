# Neuronenblitz Module Breakdown

This document lists the major logical areas within `marble_neuronenblitz/core.py`. It serves as the initial step toward refactoring the monolithic file into smaller modules.

## Learning algorithms
- Hebbian learning and synaptic echo updates
- Reinforcement and imitation learning helpers
- Curriculum and meta learning utilities

## Memory systems
- Experience replay buffer
- Episodic memory queue
- Context history tracking

## Attention and gating
- Chaotic gating logic
- Shortcut creation and usage
- Type and synapse specific attention weights

## Remote and distributed features
- Remote client and torrent offload interface
- Hooks for distributed message passing

Each of these areas can become a separate module under a future `marble_neuronenblitz` package.
