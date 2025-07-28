"""Goal Manager plugin with hierarchical goals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from reinforcement_learning import RLEnvironment


@dataclass
class Goal:
    identifier: str
    description: str
    priority: int = 0
    subgoals: List["Goal"] = field(default_factory=list)


class GoalManager:
    """Maintain goal hierarchy and resolve conflicts."""

    def __init__(self, env: Optional[RLEnvironment] = None) -> None:
        self.env = env
        self.goals: Dict[str, Goal] = {}
        self.active_goal: Optional[Goal] = None

    def add_goal(
        self,
        identifier: str,
        description: str,
        *,
        priority: int = 0,
        parent: str | None = None,
    ) -> Goal:
        goal = Goal(identifier, description, priority)
        self.goals[identifier] = goal
        if parent and parent in self.goals:
            self.goals[parent].subgoals.append(goal)
        return goal

    def choose_active_goal(self) -> Optional[Goal]:
        """Select the highest priority goal."""
        candidates = list(self.goals.values())
        if not candidates:
            return None
        candidates.sort(key=lambda g: g.priority, reverse=True)
        self.active_goal = candidates[0]
        return self.active_goal

    def shape_reward(self, reward: float) -> float:
        """Adjust reward based on the active goal."""
        if self.active_goal is None:
            self.choose_active_goal()
        if self.active_goal is None:
            return reward
        return reward + float(self.active_goal.priority)


def register(*_: object) -> None:
    """Entry point for plugin loader."""
    return
