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
                "weight_init_strategy": {"type": "string"},
                "show_message_progress": {"type": "boolean"},
                "synapse_dropout_prob": {"type": "number", "minimum": 0, "maximum": 1},
                "synapse_batchnorm_momentum": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
                "use_mixed_precision": {"type": "boolean"},
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
        "dataset": {
            "type": "object",
            "properties": {
                "num_shards": {"type": "integer", "minimum": 1},
                "shard_index": {"type": "integer", "minimum": 0},
            },
        },
        "logging": {
            "type": "object",
            "properties": {
                "structured": {"type": "boolean"},
                "log_file": {"type": ["string", "null"]},
            },
        },
        "plugins": {
            "type": ["array", "string"],
            "items": {"type": "string"},
        },
    },
    "required": ["core"],
}


def validate_config_schema(data: dict) -> None:
    """Validate configuration dict using JSON schema."""
    jsonschema.validate(instance=data, schema=CONFIG_SCHEMA)
