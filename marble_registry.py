import pickle
import tempfile
from typing import Dict, Optional

import marble_interface

class MarbleRegistry:
    """Manage multiple MARBLE instances in memory."""

    def __init__(self) -> None:
        self.instances: Dict[str, marble_interface.MARBLE] = {}
        self.active: Optional[str] = None

    def _initialize(self, cfg_path: str | None, yaml_text: str | None) -> marble_interface.MARBLE:
        if yaml_text is not None:
            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml") as tmp:
                tmp.write(yaml_text)
                cfg_path = tmp.name
        if cfg_path is None:
            raise ValueError("Either cfg_path or yaml_text must be provided")
        return marble_interface.new_marble_system(cfg_path)

    def create(
        self,
        name: str,
        cfg_path: str | None = None,
        yaml_text: str | None = None,
        *,
        overwrite: bool = False,
    ) -> marble_interface.MARBLE:
        """Create or replace a MARBLE instance named ``name``.

        Parameters
        ----------
        name:
            Identifier of the instance.
        cfg_path / yaml_text:
            Configuration for the new instance.  ``yaml_text`` takes precedence
            over ``cfg_path`` when both are provided.
        overwrite:
            When ``True`` an existing instance with the same ``name`` is
            replaced.  Otherwise the existing instance is returned unchanged.
        """

        if name in self.instances and not overwrite:
            return self.instances[name]
        marble = self._initialize(cfg_path, yaml_text)
        self.instances[name] = marble
        if self.active is None:
            self.active = name
        return marble

    def list(self) -> list[str]:
        """Return the list of registered instance names."""
        return list(self.instances.keys())

    def get(self, name: str | None = None) -> marble_interface.MARBLE:
        """Return the MARBLE instance named ``name`` or the active instance."""
        if name is None:
            name = self.active
        if name is None or name not in self.instances:
            raise ValueError("Unknown instance")
        return self.instances[name]

    def set_active(self, name: str) -> None:
        """Mark the instance named ``name`` as active."""
        if name not in self.instances:
            raise ValueError("Unknown instance")
        self.active = name

    def delete(self, name: str) -> None:
        """Remove the instance named ``name``."""
        if name not in self.instances:
            raise ValueError("Unknown instance")
        del self.instances[name]
        if self.active == name:
            self.active = next(iter(self.instances), None)

    def duplicate(self, source: str, new_name: str) -> marble_interface.MARBLE:
        """Create a deep copy of ``source`` under ``new_name``."""
        if source not in self.instances:
            raise ValueError("Unknown instance")
        if new_name in self.instances:
            raise ValueError(f"Instance {new_name!r} already exists")
        marble = pickle.loads(pickle.dumps(self.instances[source]))
        self.instances[new_name] = marble
        return marble

