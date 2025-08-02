# Python Version Compatibility

MARBLE targets Python 3.10 and newer. The library can still be transformed to run on
Python 3.8 or 3.9 using a preprocessing step, but these versions are considered
experimental.

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

The CI pipeline tests the library on Python 3.10 and 3.12 to ensure basic
compatibility. Backporting to Python 3.8/3.9 is not covered by automated tests
and may require additional validation.

## Limitations

Pattern matching and other Python 3.10 features are not supported on the older
runtimes. The conversion script does not cover every edge case so bugs may remain.
