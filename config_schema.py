import jsonschema

CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "core": {"type": "object"},
        "neuronenblitz": {"type": "object"},
        "brain": {
            "type": "object",
            "properties": {
                "early_stopping_patience": {"type": "integer", "minimum": 0},
                "early_stopping_delta": {"type": "number", "minimum": 0},
                "backup_enabled": {"type": "boolean"},
                "backup_interval": {"type": "integer", "minimum": 0},
                "backup_dir": {"type": "string"},
            },
        },
    },
    "required": ["core"],
}


def validate_config_schema(data: dict) -> None:
    """Validate configuration dict using JSON schema."""
    jsonschema.validate(instance=data, schema=CONFIG_SCHEMA)
