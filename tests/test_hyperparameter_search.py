import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hyperparameter_search import grid_search, random_search
from pipeline import Pipeline


def dummy_train(params):
    # Simple objective: minimize (a - 1)^2 + (b - 2)^2
    a = params["a"]
    b = params["b"]
    return (a - 1) ** 2 + (b - 2) ** 2


def test_grid_search_returns_sorted_results():
    grid = {"a": [0, 1], "b": [2, 3]}
    results = grid_search(grid, dummy_train)
    assert results[0][1] <= results[1][1]
    assert results[0][0] == {"a": 1, "b": 2}


def test_random_search_samples_requested_number():
    options = {"a": [0, 1, 2], "b": [1, 2, 3]}
    results = random_search(options, dummy_train, num_samples=5)
    assert len(results) == 5
    # best result should be close to dummy optimum
    best_params, best_score = results[0]
    assert best_score == min(s for _, s in results)


def test_pipeline_hyperparameter_search_grid():
    pipe = Pipeline()
    pipe.add_step(
        "scale_value",
        module="tests.dummy_pipeline_module",
        params={"a": 1.0},
        name="scale",
    )
    def score(results):
        return abs(results[0] - 3.0)
    results = pipe.hyperparameter_search({"scale.scale": [1.0, 2.0, 3.0]}, score)
    best_params, best_score = results[0]
    assert best_params["scale.scale"] == 3.0
    assert best_score == 0.0


def test_pipeline_hyperparameter_search_random():
    pipe = Pipeline()
    pipe.add_step(
        "scale_value",
        module="tests.dummy_pipeline_module",
        params={"a": 1.0},
        name="scale",
    )
    def score(results):
        return abs(results[0] - 3.0)
    results = pipe.hyperparameter_search(
        {"scale.scale": [1.0, 2.0, 3.0]},
        score,
        search="random",
        num_samples=2,
    )
    assert len(results) == 2
