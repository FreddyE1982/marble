from setuptools import setup

setup(
    name="marble",
    version="0.1.0",
    py_modules=[
        "marble_core",
        "marble_neuronenblitz",
        "marble",
        "marble_brain",
        "reinforcement_learning",
        "adversarial_utils",
        "adversarial_dataset_wrapper",
        "self_monitoring_plugin",
        "context_history",
    ],
    install_requires=[],
)
