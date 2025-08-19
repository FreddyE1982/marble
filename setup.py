from pathlib import Path

from setuptools import find_packages, setup

root = Path(__file__).parent
py_modules = [
    p.stem for p in root.glob("*.py") if p.name not in {"setup.py", "install.py"}
]

setup(
    name="marble",
    version="0.1.0",
    packages=find_packages(),
    py_modules=py_modules,
    install_requires=[],
)
