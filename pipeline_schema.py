"""JSON schema for validating pipeline step configurations.

Each step must define a ``name`` and either a ``func``, ``plugin`` or
``branches`` entry.  The schema is recursive so nested branches are validated
as well.  :func:`validate_step_schema` raises ``jsonschema.ValidationError`` if
a step does not conform to the expected structure.
"""

from __future__ import annotations

import jsonschema

STEP_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "func": {"type": "string"},
        "module": {"type": ["string", "null"]},
        "params": {"type": "object"},
        "plugin": {"type": "string"},
        "depends_on": {"type": "array", "items": {"type": "string"}},
        "tier": {"type": "string"},
        "memory_limit_mb": {"type": ["number", "null"], "minimum": 0},
        "branches": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"$ref": "#"},
            },
        },
        "merge": {"type": "object"},
        "macro": {
            "type": "array",
            "items": {"$ref": "#"},
        },
    },
    "required": ["name"],
    "anyOf": [
        {"required": ["func"]},
        {"required": ["plugin"]},
        {"required": ["branches"]},
        {"required": ["macro"]},
    ],
}


def validate_step_schema(step: dict) -> None:
    """Validate a pipeline step against :data:`STEP_SCHEMA`.

    Parameters
    ----------
    step:
        Step specification dictionary to validate.
    """
    jsonschema.validate(instance=step, schema=STEP_SCHEMA)
