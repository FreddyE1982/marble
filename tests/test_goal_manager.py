import importlib

import goal_manager


def test_goal_manager_selects_highest_priority():
    importlib.reload(goal_manager)
    gm = goal_manager.GoalManager()
    gm.add_goal("g1", "first", priority=1)
    gm.add_goal("g2", "second", priority=5)
    active = gm.choose_active_goal()
    assert active.identifier == "g2"
    shaped = gm.shape_reward(1.0)
    assert shaped > 1.0
