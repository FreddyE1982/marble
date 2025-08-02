# Python Compatibility Notes

The codebase targets Python 3.10 and higher. Several language features used in the implementation
require shims on older versions:

- **Union types and builtin generics** – Modules using ``int | None`` or ``list[str]`` include
  ``from __future__ import annotations`` so that these annotations are parsed correctly on Python 3.10+ and
  can be backported to Python 3.8/3.9 when needed.
- **``str.removeprefix``** – Provided via :func:`pycompat.removeprefix` for older runtimes.
- **``functools.cache``** – Implemented as :func:`pycompat.cached` when running on Python 3.8.

Continuous integration runs the full test suite on Python 3.10 and 3.12 to guarantee that new
changes remain compatible. The `scripts/convert_to_py38.py` tool can be used to backport the
code to Python 3.8/3.9, but these runtimes are no longer officially supported.
The `pycompat` module centralises these shims so components like
`remote_offload` and `dataset_cache_server` remain usable on older interpreters.
