import jsonschema

CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "core": {
            "type": "object",
            "properties": {
                "backend": {"type": "string"},
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
                "default_growth_tier": {"type": "string"},
                "pretraining_epochs": {"type": "integer", "minimum": 0},
                "min_cluster_k": {"type": "integer", "minimum": 1},
                "attention_gating": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "mode": {"type": "string"},
                        "frequency": {"type": "number", "minimum": 0},
                        "chaos": {"type": "number", "minimum": 0, "maximum": 4},
                    },
                },
                "interconnection_prob": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
            },
        },
        "neuronenblitz": {"type": "object"},
        "sac": {
            "type": "object",
            "properties": {
                "temperature": {"type": "number", "minimum": 0},
            },
        },
        "evolution": {
            "type": "object",
            "properties": {
                "population_size": {"type": "integer", "minimum": 1},
                "selection_size": {"type": "integer", "minimum": 1},
                "generations": {"type": "integer", "minimum": 1},
                "steps_per_candidate": {"type": "integer", "minimum": 1},
                "mutation_rate": {"type": "number", "minimum": 0, "maximum": 1},
                "parallelism": {"type": "integer", "minimum": 1},
            },
        },
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
                "encryption_enabled": {"type": "boolean"},
                "encryption_key": {"type": ["string", "null"]},
                "source": {"type": ["string", "null"]},
                "cache_url": {"type": ["string", "null"]},
                "use_kuzu_graph": {"type": "boolean"},
                "version_registry": {"type": ["string", "null"]},
                "version": {"type": ["string", "null"]},
                "kuzu_graph": {
                    "type": "object",
                    "properties": {
                        "db_path": {"type": "string"},
                        "query": {"type": "string"},
                        "input_column": {"type": "string"},
                        "target_column": {"type": "string"},
                        "limit": {"type": ["integer", "null"], "minimum": 1},
                    },
                },
            },
        },
        "logging": {
            "type": "object",
            "properties": {
                "structured": {"type": "boolean"},
                "log_file": {"type": ["string", "null"]},
                "level": {"type": ["string", "integer"]},
                "format": {"type": ["string", "null"]},
                "datefmt": {"type": ["string", "null"]},
                "propagate": {"type": "boolean"},
                "rotate": {"type": "boolean"},
                "max_bytes": {"type": "integer", "minimum": 1},
                "backup_count": {"type": "integer", "minimum": 1},
                "encoding": {"type": ["string", "null"]},
            },
        },
        "predictive_coding": {
            "type": "object",
            "properties": {
                "num_layers": {"type": "integer", "minimum": 1},
                "latent_dim": {"type": "integer", "minimum": 1},
                "learning_rate": {"type": "number", "minimum": 0},
            },
        },
        "pipeline": {
            "type": "object",
            "properties": {
                "default_step_memory_limit_mb": {
                    "type": ["number", "null"],
                    "minimum": 0,
                }
            },
        },
        "scheduler": {
            "type": "object",
            "properties": {"plugin": {"type": "string"}},
        },
        "data_compressor": {
            "type": "object",
            "properties": {
                "compression_level": {"type": "integer", "minimum": 0, "maximum": 9},
                "compression_enabled": {"type": "boolean"},
                "delta_encoding": {"type": "boolean"},
                "compression_algorithm": {"type": "string", "enum": ["zlib", "lzma"]},
                "quantization_bits": {"type": "integer", "minimum": 0, "maximum": 8},
                "sparse_threshold": {
                    "type": ["number", "null"],
                    "minimum": 0,
                    "maximum": 1,
                },
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
        "mcp_server": {
            "type": "object",
            "properties": {
                "host": {"type": "string"},
                "port": {"type": "integer", "minimum": 1, "maximum": 65535},
                "auth": {
                    "type": "object",
                    "properties": {
                        "token": {"type": ["string", "null"]},
                        "username": {"type": ["string", "null"]},
                        "password": {"type": ["string", "null"]},
                    },
                },
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
        "live_kuzu": {
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
                        "enabled": {"type": "boolean"},
                        "host": {"type": "string"},
                        "port": {"type": "integer", "minimum": 1},
                        "remote_url": {"type": ["string", "null"]},
                        "auth_token": {"type": ["string", "null"]},
                        "ssl_enabled": {"type": "boolean"},
                        "ssl_cert_file": {"type": ["string", "null"]},
                        "ssl_key_file": {"type": ["string", "null"]},
                        "compression_level": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 9,
                        },
                        "compression_enabled": {"type": "boolean"},
                    },
                },
                "remote_client": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "timeout": {"type": "number", "minimum": 0},
                        "max_retries": {"type": "integer", "minimum": 0},
                        "backoff_factor": {"type": "number", "minimum": 0},
                        "track_latency": {"type": "boolean"},
                        "auth_token": {"type": ["string", "null"]},
                        "ssl_verify": {"type": "boolean"},
                        "connect_retry_interval": {"type": "number", "minimum": 0},
                        "heartbeat_timeout": {"type": "number", "minimum": 0},
                        "use_compression": {"type": "boolean"},
                    },
                },
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
