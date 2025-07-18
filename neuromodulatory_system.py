class NeuromodulatorySystem:
    """Tracks neuromodulatory signals like arousal, stress and reward."""

    def __init__(self, initial=None):
        self.signals = {
            "arousal": 0.0,
            "stress": 0.0,
            "reward": 0.0,
            "emotion": "neutral",
        }
        if initial:
            self.update_signals(**initial)

    def update_signals(self, **kwargs):
        for key, value in kwargs.items():
            self.signals[key] = value

    def get_context(self):
        return self.signals.copy()
