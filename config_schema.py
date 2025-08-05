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
                "quantization_bits": {"type": "integer", "minimum": 0, "maximum": 16},
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
                "checkpoint_format": {"type": "string"},
                "checkpoint_compress": {"type": "boolean"},
                "profile_interval": {"type": "integer", "minimum": 1},
            },
        },
        "dataset": {
            "type": "object",
            "properties": {
                "num_shards": {"type": "integer", "minimum": 1},
                "shard_index": {"type": "integer", "minimum": 0},
                "offline": {"type": "boolean"},
                "encryption_key": {"type": ["string", "null"]},
            },
        },
        "logging": {
            "type": "object",
            "properties": {
                "structured": {"type": "boolean"},
                "log_file": {"type": ["string", "null"]},
            },
        },
        "data_compressor": {
            "type": "object",
            "properties": {
                "compression_level": {"type": "integer", "minimum": 0, "maximum": 9},
                "compression_enabled": {"type": "boolean"},
                "delta_encoding": {"type": "boolean"},
                "compression_algorithm": {"type": "string", "enum": ["zlib", "lzma"]},
            },
        },
        "dataloader": {
            "type": "object",
            "properties": {
                "tensor_dtype": {"type": "string"},
                "track_metadata": {"type": "boolean"},
                "enable_round_trip_check": {"type": "boolean"},
                "round_trip_penalty": {"type": "number", "minimum": 0},
                "tokenizer_type": {"type": ["string", "null"]},
                "tokenizer_json": {"type": ["string", "null"]},
                "tokenizer_vocab_size": {"type": "integer", "minimum": 1},
            },
        },
        "sync": {
            "type": "object",
            "properties": {
                "interval_ms": {"type": "integer", "minimum": 1},
            },
        },
        "plugins": {
            "type": ["array", "string"],
            "items": {"type": "string"},
        },
        "remote_hardware": {
            "type": "object",
            "properties": {
                "tier_plugin": {"type": ["string", "null"]},
                "grpc": {
                    "type": "object",
                    "properties": {
                        "address": {"type": "string"},
                    },
                },
            },
        },
        "topology_graph": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "db_path": {"type": "string"},
            },
        },
        "network": {
            "type": "object",
            "properties": {
                "remote_server": {
                    "type": "object",
                    "properties": {
                        "auth_token": {"type": ["string", "null"]},
                    },
                },
                "remote_client": {"type": "object"},
                "torrent_client": {"type": "object"},
            },
        },
        "autograd": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "learning_rate": {"type": "number"},
                "gradient_accumulation_steps": {"type": "integer", "minimum": 1},
            },
        },
        "global_workspace": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "capacity": {"type": "integer", "minimum": 1},
            },
        },
        "experiments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "core": {"type": "object"},
                    "neuronenblitz": {"type": "object"},
                    "brain": {"type": "object"},
                },
                "required": ["name"],
            },
        },
    },
    "required": ["core"],
}


def validate_config_schema(data: dict) -> None:
    """Validate configuration dict using JSON schema."""
    jsonschema.validate(instance=data, schema=CONFIG_SCHEMA)
