from context_history import ContextEntry


def test_context_entry_fields():
    entry = ContextEntry({'arousal': 1.0})
    assert entry.context['arousal'] == 1.0
    assert entry.markers == []
    assert entry.goals == []
