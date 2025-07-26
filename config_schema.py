import jsonschema

CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "core": {
            "type": "object",
            "properties": {
                "representation_size": {"type": "integer", "minimum": 1},
                "message_passing_alpha": {"type": "number", "minimum": 0, "maximum": 1},
                "message_passing_beta": {"type": "number", "minimum": 0, "maximum": 1},
            },
        },
        "neuronenblitz": {"type": "object"},
        "brain": {
            "type": "object",
            "properties": {
                "early_stopping_patience": {"type": "integer", "minimum": 0},
                "early_stopping_delta": {"type": "number", "minimum": 0},
                "backup_enabled": {"type": "boolean"},
                "backup_interval": {"type": "integer", "minimum": 0},
                "backup_dir": {"type": "string"},
                "profile_enabled": {"type": "boolean"},
                "profile_log_path": {"type": "string"},
                "profile_interval": {"type": "integer", "minimum": 1},
            },
        },
    },
    "required": ["core"],
}


def validate_config_schema(data: dict) -> None:
    """Validate configuration dict using JSON schema."""
    jsonschema.validate(instance=data, schema=CONFIG_SCHEMA)
