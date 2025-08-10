import tensor_backend as tb
from examples import dynamic_message_passing_example as dmpe


def test_dynamic_message_passing_example_runs():
    (reps1, reps2, reps3), syn_counts = dmpe.run_demo(return_history=True)
    assert syn_counts == [1, 2, 1]
    xp = tb.xp()
    diff_add = float(tb.to_numpy(xp.linalg.norm(xp.array(reps2[0]) - xp.array(reps1[0]))))
    assert diff_add > 0.0
