from pytorch_challenge import (
    load_pretrained_model,
    load_dataset,
    train_marble_with_challenge,
)


def test_dataset_size():
    data = load_dataset(10)
    assert len(data) == 10
    assert data[0][0].shape == (8, 8)


def test_pretrained_model_loads():
    model = load_pretrained_model()
    assert hasattr(model, "forward")


def test_challenge_training_improves_over_pytorch():
    data = load_dataset(100)
    train = data[:80]
    val = data[80:]
    model = load_pretrained_model()
    results = train_marble_with_challenge(train, val, model, epochs=10, seed=0)
    assert results["marble"]["loss"] <= results["pytorch"]["loss"]
    assert results["marble"]["time"] <= results["pytorch"]["time"]
    assert results["marble"]["size"] <= results["pytorch"]["size"]
