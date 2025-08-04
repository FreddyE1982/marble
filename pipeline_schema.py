from __future__ import annotations

"""JSON schema for validating pipeline step configurations.

Each step must define a ``name`` and either a ``func``, ``plugin`` or
``branches`` entry.  The schema is recursive so nested branches are validated
as well.  :func:`validate_step_schema` raises ``jsonschema.ValidationError`` if
a step does not conform to the expected structure.
"""

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
        "branches": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"$ref": "#"},
            },
        },
        "merge": {"type": "object"},
    },
    "required": ["name"],
    "anyOf": [
        {"required": ["func"]},
        {"required": ["plugin"]},
        {"required": ["branches"]},
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
