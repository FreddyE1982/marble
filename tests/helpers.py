values = []


def append_value(value: str, store=None):
    """Append ``value`` to ``store`` or a module-level list."""
    target = store if store is not None else values
    target.append(value)
    return target
