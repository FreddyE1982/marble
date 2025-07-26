import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hyperparameter_search import grid_search, random_search


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
