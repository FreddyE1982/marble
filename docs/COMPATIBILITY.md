# Python Version Compatibility

MARBLE primarily targets Python 3.11 but can run on Python 3.8 and 3.9 with a preprocessing step.

## Incompatible Features

- **PEP 604** union syntax (e.g. `int | None`) is used throughout the code base. Python <3.10 cannot parse this syntax.
- Built-in generics like `list[str]` require Python 3.9 or newer.

## Polyfill Script

The `scripts/convert_to_py38.py` tool rewrites source files to use `typing.Union`
and `typing.List` so the code can execute on Python 3.8/3.9. Run it before
installation:

```bash
python scripts/convert_to_py38.py path/to/marble
```

## Continuous Integration

The CI pipeline tests the library on Python 3.8, 3.9 and 3.11 to ensure basic
compatibility. Some optional features may be disabled on older versions.

## Limitations

Pattern matching and other Python 3.10 features are not supported on the older
runtimes. The conversion script does not cover every edge case so bugs may remain.
