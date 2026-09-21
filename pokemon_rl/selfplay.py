"""Episode runner shared by training and evaluation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import numpy as np

from .agents import DQNAgent, GreedyAgent, Transition
from .battle import N_ACTIONS, SIDES, Battle


class Policy(Protocol):
    def act(self, state: np.ndarray, mask: np.ndarray, greedy: bool = False) -> int: ...


@dataclass
class EpisodeResult:
    winner: int | None
    turns: int
    decisions: int
    log: list[str]


def play_episode(
    policies: Sequence[Policy],
    rng: random.Random,
    learner: DQNAgent | None = None,
    learner_sides: Sequence[int] = (),
    shaping: float = 0.0,
    greedy: bool = False,
    max_turns: int = 200,
    on_transition: Callable[[], None] | None = None,
) -> EpisodeResult:
    """Run one battle. Transitions of `learner_sides` are pushed to `learner`.

    A side's transition spans from the moment it picks an action to its next
    decision point, which may be several battle steps later (the opponent can
    be replacing a fainted Pokemon meanwhile).  The intermediate HP swing is
    folded into the shaping reward, so nothing is lost by the gap.
    """
    battle = Battle(rng=rng, max_turns=max_turns)
    for side, policy in enumerate(policies):
        if isinstance(policy, GreedyAgent):
            policy.attach(battle, side)

    pending: dict[int, tuple[np.ndarray, int, float] | None] = {0: None, 1: None}
    decisions = 0

    def push(side: int, next_state: np.ndarray, next_mask: np.ndarray, done: bool,
             terminal_reward: float = 0.0) -> None:
        entry = pending[side]
        if learner is None or entry is None:
            return
        state, action, hp_diff_before = entry
        reward = shaping * (battle.hp_diff(side) - hp_diff_before) + terminal_reward
        learner.remember(Transition(state, action, reward, next_state, next_mask, done))
        pending[side] = None
        if on_transition is not None:
            on_transition()

    while not battle.done:
        actions: dict[int, int] = {}
        acting: list[tuple[int, np.ndarray, np.ndarray, int]] = []
        for side in SIDES:
            if battle.needs_action(side):
                state = battle.observation(side)
                mask = battle.legal_mask(side)
                action = policies[side].act(state, mask, greedy=greedy)
                actions[side] = action
                acting.append((side, state, mask, action))
                decisions += 1

        for side, state, mask, action in acting:
            if learner is not None and side in learner_sides and pending[side] is not None:
                push(side, state, mask, done=False)
            pending[side] = (state, action, battle.hp_diff(side))

        battle.step(actions)

    if learner is not None:
        empty_mask = np.zeros(N_ACTIONS, dtype=bool)
        for side in learner_sides:
            if pending[side] is not None:
                outcome = 0.0 if battle.winner is None else (1.0 if battle.winner == side else -1.0)
                push(side, battle.observation(side), empty_mask, done=True, terminal_reward=outcome)

    return EpisodeResult(battle.winner, battle.turns_played, decisions, battle.log)
