class DotDict(dict):
    """Dictionary supporting attribute-style access to keys."""

    def __init__(self, *args, **kwargs):
        data = dict(*args, **kwargs)
        for k, v in data.items():
            if isinstance(v, dict) and not isinstance(v, DotDict):
                data[k] = DotDict(v)
        super().__init__(data)

    def __getattr__(self, name):
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        if isinstance(value, dict) and not isinstance(value, DotDict):
            value = DotDict(value)
            self[name] = value
        return value

    def __setattr__(self, name, value):
        if name.startswith('_'):
            return super().__setattr__(name, value)
        if isinstance(value, dict) and not isinstance(value, DotDict):
            value = DotDict(value)
        self[name] = value

    def update(self, *args, **kwargs):
        data = dict(*args, **kwargs)
        for k, v in data.items():
            if isinstance(v, dict) and not isinstance(v, DotDict):
                v = DotDict(v)
            self[k] = v

    def copy(self):
        return DotDict(super().copy())

    def to_dict(self):
        out = {}
        for k, v in self.items():
            if isinstance(v, DotDict):
                out[k] = v.to_dict()
            else:
                out[k] = v
        return out
