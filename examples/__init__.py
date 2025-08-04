import os

# Disable metrics visualisation for example runs to keep them fast and
# avoid heavy matplotlib usage during automated execution environments.
os.environ.setdefault("MARBLE_DISABLE_METRICS", "1")

