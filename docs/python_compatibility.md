# Python Compatibility Notes

The codebase targets Python 3.8 and higher. Several language features used in the implementation
require shims on older versions:

- **Union types and builtin generics** – Modules using ``int | None`` or ``list[str]`` now include
  ``from __future__ import annotations`` so that these annotations are parsed correctly on Python 3.8/3.9.
- **``str.removeprefix``** – Provided via :func:`pycompat.removeprefix` for older runtimes.
- **``functools.cache``** – Implemented as :func:`pycompat.cached` when running on Python 3.8.

Continuous integration runs the full test suite on Python 3.8, 3.9 and 3.11 to guarantee that new
changes remain compatible. Feature parity is maintained, but performance may vary on older versions.
