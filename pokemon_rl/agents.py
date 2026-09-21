"""Deep Q-learning agent (scikit-learn MLPRegressor) and scripted baselines."""

from __future__ import annotations

import copy
import pickle
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPRegressor

from .battle import Battle, MOVE_ACTIONS, N_ACTIONS, OBS_SIZE, PokemonState, compute_damage


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    next_mask: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, rng: random.Random) -> None:
        self.buffer: deque[Transition] = deque(maxlen=capacity)
        self.rng = rng

    def add(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return self.rng.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


def _masked_argmax(values: np.ndarray, mask: np.ndarray) -> int:
    masked = np.where(mask, values, -np.inf)
    return int(np.argmax(masked))


class DQNAgent:
    """Q-network over the 5 actions, trained with replay + a target network.

    The network is a scikit-learn MLPRegressor with 5 outputs; each SGD update
    is one `partial_fit` on a replay batch where only the taken action's target
    differs from the current prediction.
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (128, 128),
        learning_rate: float = 1e-3,
        gamma: float = 0.98,
        buffer_size: int = 50_000,
        batch_size: int = 64,
        warmup: int = 1_000,
        target_sync: int = 500,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 20_000,
        seed: int | None = None,
    ) -> None:
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup = warmup
        self.target_sync = target_sync
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.rng = random.Random(seed)
        self.buffer = ReplayBuffer(buffer_size, self.rng)
        self.train_steps = 0

        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            learning_rate_init=learning_rate,
            random_state=seed,
        )
        # One dummy update so the weights (and the output size) exist before the
        # first real prediction; the target network then mirrors them.
        self.model.partial_fit(np.zeros((1, OBS_SIZE)), np.zeros((1, N_ACTIONS)))
        self.target_model = copy.deepcopy(self.model)

    # ----------------------------------------------------------------- policy

    @property
    def epsilon(self) -> float:
        progress = min(1.0, self.train_steps / max(1, self.epsilon_decay_steps))
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def q_values(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(state.reshape(1, -1)))[0]

    def act(self, state: np.ndarray, mask: np.ndarray, greedy: bool = False) -> int:
        legal = np.flatnonzero(mask)
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.choice(legal.tolist()))
        return _masked_argmax(self.q_values(state), mask)

    # --------------------------------------------------------------- learning

    def remember(self, transition: Transition) -> None:
        self.buffer.add(transition)

    def train_step(self) -> float | None:
        """One gradient step on a replay batch; returns the batch TD loss."""
        if len(self.buffer) < max(self.warmup, self.batch_size):
            return None

        batch = self.buffer.sample(self.batch_size)
        states = np.stack([t.state for t in batch])
        next_states = np.stack([t.next_state for t in batch])
        next_masks = np.stack([t.next_mask for t in batch])

        next_q = np.asarray(self.target_model.predict(next_states))
        next_q = np.where(next_masks, next_q, -np.inf)
        best_next = np.where(next_masks.any(axis=1), next_q.max(axis=1), 0.0)

        rewards = np.array([t.reward for t in batch])
        dones = np.array([t.done for t in batch], dtype=bool)
        targets_for_action = rewards + np.where(dones, 0.0, self.gamma * best_next)

        targets = np.asarray(self.model.predict(states))
        actions = np.array([t.action for t in batch])
        rows = np.arange(len(batch))
        loss = float(np.mean((targets[rows, actions] - targets_for_action) ** 2))
        targets[rows, actions] = targets_for_action

        self.model.partial_fit(states, targets)
        self.train_steps += 1
        if self.train_steps % self.target_sync == 0:
            self.target_model = copy.deepcopy(self.model)
        return loss

    def snapshot(self, epsilon: float = 0.05, seed: int | None = None) -> "FrozenPolicy":
        """Freeze the current weights into a standalone opponent policy."""
        return FrozenPolicy(copy.deepcopy(self.model), epsilon=epsilon, seed=seed)

    # ------------------------------------------------------------ persistence

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as handle:
            pickle.dump({"model": self.model, "train_steps": self.train_steps}, handle)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        self.model = payload["model"]
        self.train_steps = payload["train_steps"]
        self.target_model = copy.deepcopy(self.model)


class FrozenPolicy:
    """A frozen copy of a Q-network, used as a fixed self-play opponent.

    Training only against the current policy lets the agent chase itself around
    the matchup triangle; keeping a pool of past snapshots around keeps it
    honest against strategies it used to face.
    """

    name = "snapshot"

    def __init__(self, model: MLPRegressor, epsilon: float = 0.05,
                 seed: int | None = None) -> None:
        self.model = model
        self.epsilon = epsilon
        self.rng = random.Random(seed)

    def act(self, state: np.ndarray, mask: np.ndarray, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.choice(np.flatnonzero(mask).tolist()))
        q_values = np.asarray(self.model.predict(state.reshape(1, -1)))[0]
        return _masked_argmax(q_values, mask)


class RandomAgent:
    """Uniformly random legal action."""

    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def act(self, state: np.ndarray, mask: np.ndarray, greedy: bool = True) -> int:
        return int(self.rng.choice(np.flatnonzero(mask).tolist()))


class GreedyAgent:
    """Scripted opponent: always fire the hardest-hitting move, never switch
    voluntarily; on a forced switch pick the best type matchup.

    Unlike the learned policies this one reads the battle directly rather than
    the observation vector, so it has to be attached to the battle it plays.
    """

    name = "greedy"

    def __init__(self, seed: int | None = None) -> None:
        self.battle: Battle | None = None
        self.side = 0
        self.rng = random.Random(seed)

    def attach(self, battle: Battle, side: int) -> None:
        self.battle, self.side = battle, side

    def act(self, state: np.ndarray, mask: np.ndarray, greedy: bool = True) -> int:
        if self.battle is None:
            raise RuntimeError("attach the agent to a battle before asking it to act")
        battle, side = self.battle, self.side
        legal = np.flatnonzero(mask).tolist()
        attacker = battle.active_pokemon(side)
        defender = battle.active_pokemon(1 - side)

        if attacker is not None and defender is not None and any(a < MOVE_ACTIONS for a in legal):
            damages = [compute_damage(attacker, defender, move)
                       for move in attacker.species.moves]
            return int(np.argmax(damages))

        switches = [a for a in legal if a >= MOVE_ACTIONS]
        return self._best_switch(battle, side, defender, switches)

    def _best_switch(self, battle: Battle, side: int, defender: PokemonState | None,
                     switches: list[int]) -> int:
        if defender is None:
            return int(self.rng.choice(switches))

        def score(action: int) -> float:
            """Damage dealt minus damage taken, both as a share of max HP."""
            candidate = battle.team[side][action - MOVE_ACTIONS]
            offense = max(compute_damage(candidate, defender, move) / defender.species.hp
                          for move in candidate.species.moves)
            incoming = max(compute_damage(defender, candidate, move) / candidate.species.hp
                           for move in defender.species.moves)
            return offense - incoming

        return max(switches, key=score)
