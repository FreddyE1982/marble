# Public API Overview

This document lists the main classes and functions intended for external use.

- `marble_core.Core`
- `marble_neuronenblitz.Neuronenblitz`
- `marble_neuronenblitz.learning.enable_rl`
- `marble_neuronenblitz.learning.disable_rl`
- `marble_neuronenblitz.learning.rl_select_action`
- `marble_neuronenblitz.learning.rl_update`
- `marble_neuronenblitz.memory.decay_memory_gates`
- `global_workspace.activate`
- `global_workspace.GlobalWorkspace.publish`
- `global_workspace.GlobalWorkspace.subscribe`
- `attention_codelets.register_codelet`
- `attention_codelets.run_cycle`

These APIs are kept stable across minor versions. Internal helpers not listed here may change without notice.
