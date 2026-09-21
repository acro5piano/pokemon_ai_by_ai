"""A minimal Generation-1 battle engine.

Simplifications (as specified):
  * every move hits (100% accuracy), no critical hits, no status moves,
    no secondary effects and no PP limits;
  * teams are the same fixed three Pokemon, only the order of play is chosen;
  * damage keeps the Gen-1 random roll, so battles stay stochastic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from .data import LEVEL, PHYSICAL, SPECIES, TEAM, TEAM_SIZE, Move, Species, type_effectiveness

# Action space: two moves of the active Pokemon, then one switch per team slot.
MOVE_ACTIONS = 2
N_ACTIONS = MOVE_ACTIONS + TEAM_SIZE  # 5
OBS_SIZE = 2 * (3 * TEAM_SIZE) + 1  # per side: hp / alive / active, plus a phase flag

PHASE_LEAD = "lead"
PHASE_MOVE = "move"
PHASE_REPLACE = "replace"
PHASE_END = "end"

SIDES = (0, 1)


@dataclass
class PokemonState:
    species: Species
    hp: int = field(init=False)

    def __post_init__(self) -> None:
        self.hp = self.species.hp

    @property
    def fainted(self) -> bool:
        return self.hp <= 0

    @property
    def hp_fraction(self) -> float:
        return max(0.0, self.hp / self.species.hp)

    def __str__(self) -> str:
        return f"{self.species.name}({self.hp}/{self.species.hp})"


def compute_damage(
    attacker: PokemonState,
    defender: PokemonState,
    move: Move,
    roll: int = 255,
) -> int:
    """Gen-1 damage formula without crits, with the 217..255 damage roll."""
    effectiveness = type_effectiveness(move.type, defender.species.types)
    if effectiveness == 0.0:
        return 0

    if move.category == PHYSICAL:
        attack, defense = attacker.species.attack, defender.species.defense
    else:  # Gen 1 uses the single Special stat on both sides.
        attack, defense = attacker.species.special, defender.species.special

    damage = ((2 * LEVEL // 5 + 2) * move.power * attack // defense) // 50 + 2
    if move.type in attacker.species.types:
        damage = damage * 3 // 2  # STAB
    damage = int(damage * effectiveness)
    damage = damage * roll // 255
    return max(1, damage)


class Battle:
    """Two-sided battle driven by simultaneous action selection."""

    def __init__(self, rng: random.Random | None = None, max_turns: int = 200) -> None:
        self.rng = rng or random.Random()
        self.max_turns = max_turns
        self.team: list[list[PokemonState]] = [
            [PokemonState(SPECIES[name]) for name in TEAM] for _ in SIDES
        ]
        self.active: list[int | None] = [None, None]
        self.phase = PHASE_LEAD
        self.turn = 0
        self.winner: int | None = None
        self.log: list[str] = []

    # ------------------------------------------------------------------ state

    @property
    def done(self) -> bool:
        return self.phase == PHASE_END

    @property
    def turns_played(self) -> int:
        return self.turn

    def active_pokemon(self, side: int) -> PokemonState | None:
        slot = self.active[side]
        return None if slot is None else self.team[side][slot]

    def require_active(self, side: int) -> PokemonState:
        """The active Pokemon of a side that is known to have one."""
        mon = self.active_pokemon(side)
        if mon is None:
            raise RuntimeError(f"side {side} has no active Pokemon yet")
        return mon

    def alive_slots(self, side: int) -> list[int]:
        return [i for i, mon in enumerate(self.team[side]) if not mon.fainted]

    def needs_action(self, side: int) -> bool:
        if self.phase in (PHASE_LEAD, PHASE_MOVE):
            return True
        if self.phase == PHASE_REPLACE:
            mon = self.active_pokemon(side)
            return mon is None or mon.fainted
        return False

    def legal_actions(self, side: int) -> list[int]:
        if self.phase in (PHASE_LEAD, PHASE_REPLACE):
            return [MOVE_ACTIONS + slot for slot in self.alive_slots(side)
                    if slot != self.active[side]]
        if self.phase == PHASE_MOVE:
            switches = [MOVE_ACTIONS + slot for slot in self.alive_slots(side)
                        if slot != self.active[side]]
            return list(range(MOVE_ACTIONS)) + switches
        return []

    def legal_mask(self, side: int) -> np.ndarray:
        mask = np.zeros(N_ACTIONS, dtype=bool)
        mask[self.legal_actions(side)] = True
        return mask

    def observation(self, side: int) -> np.ndarray:
        """Battle state from `side`'s point of view (own team first)."""
        features: list[float] = []
        for owner in (side, 1 - side):
            for slot, mon in enumerate(self.team[owner]):
                features.append(mon.hp_fraction)
                features.append(0.0 if mon.fainted else 1.0)
                features.append(1.0 if self.active[owner] == slot else 0.0)
        features.append(1.0 if self.phase in (PHASE_LEAD, PHASE_REPLACE) else 0.0)
        return np.asarray(features, dtype=np.float64)

    def hp_diff(self, side: int) -> float:
        """Own remaining HP minus the opponent's, in team fractions (-1..1)."""
        mine = sum(mon.hp_fraction for mon in self.team[side])
        theirs = sum(mon.hp_fraction for mon in self.team[1 - side])
        return (mine - theirs) / TEAM_SIZE

    # ------------------------------------------------------------- transition

    def step(self, actions: dict[int, int]) -> None:
        if self.done:
            raise RuntimeError("battle is over")
        for side in SIDES:
            if self.needs_action(side):
                if side not in actions:
                    raise ValueError(f"side {side} must act during phase {self.phase}")
                if actions[side] not in self.legal_actions(side):
                    raise ValueError(f"illegal action {actions[side]} for side {side}")

        if self.phase == PHASE_LEAD:
            self._switch_phase(actions, "leads with")
            self.phase = PHASE_MOVE
        elif self.phase == PHASE_REPLACE:
            self._switch_phase(actions, "sends out")
            self.phase = PHASE_MOVE
        else:
            self._resolve_turn(actions)
            self._check_end()

    def _switch_phase(self, actions: dict[int, int], verb: str) -> None:
        for side in SIDES:
            if side in actions and self.needs_action(side):
                slot = actions[side] - MOVE_ACTIONS
                self.active[side] = slot
                self._log(f"P{side + 1} {verb} {self.team[side][slot].species.name}")

    def _resolve_turn(self, actions: dict[int, int]) -> None:
        self.turn += 1
        self._log(f"-- turn {self.turn} --")

        for side in SIDES:
            if actions[side] >= MOVE_ACTIONS:
                slot = actions[side] - MOVE_ACTIONS
                self.active[side] = slot
                self._log(f"P{side + 1} switches to {self.team[side][slot].species.name}")

        attackers = [side for side in SIDES if actions[side] < MOVE_ACTIONS]
        for side in self._move_order(attackers):
            if self.require_active(side).fainted:
                continue
            self._use_move(side, actions[side])

    def _move_order(self, attackers: list[int]) -> list[int]:
        if len(attackers) < 2:
            return attackers
        speeds = [self.require_active(side).species.speed for side in attackers]
        if speeds[0] == speeds[1]:
            first = self.rng.choice(attackers)
            return [first, 1 - first]
        return sorted(attackers, key=lambda s: -self.require_active(s).species.speed)

    def _use_move(self, side: int, move_index: int) -> None:
        attacker = self.require_active(side)
        defender = self.require_active(1 - side)
        move = attacker.species.moves[move_index]
        roll = self.rng.randint(217, 255)
        damage = compute_damage(attacker, defender, move, roll)
        effectiveness = type_effectiveness(move.type, defender.species.types)

        if effectiveness == 0.0:
            self._log(f"P{side + 1} {attacker.species.name} used {move.name}: "
                      f"it doesn't affect P{2 - side}'s {defender.species.name}")
            return

        defender.hp = max(0, defender.hp - damage)
        note = ""
        if effectiveness > 1.0:
            note = " (super effective)"
        elif effectiveness < 1.0:
            note = " (not very effective)"
        self._log(f"P{side + 1} {attacker.species.name} used {move.name}{note}: "
                  f"{damage} damage, P{2 - side}'s {defender.species.name} at "
                  f"{defender.hp}/{defender.species.hp}")
        if defender.fainted:
            self._log(f"P{2 - side}'s {defender.species.name} fainted")

    def _check_end(self) -> None:
        alive = [self.alive_slots(side) for side in SIDES]
        if not alive[0] or not alive[1]:
            self.phase = PHASE_END
            if alive[0] and not alive[1]:
                self.winner = 0
            elif alive[1] and not alive[0]:
                self.winner = 1
            else:
                self.winner = None
            result = "draw" if self.winner is None else f"P{self.winner + 1} wins"
            self._log(f"== battle over after {self.turn} turns: {result} ==")
            return

        if self.turn >= self.max_turns:
            self.phase = PHASE_END
            self.winner = None
            self._log(f"== battle over: turn limit ({self.max_turns}) reached, draw ==")
            return

        if any(self.require_active(side).fainted for side in SIDES):
            self.phase = PHASE_REPLACE

    def _log(self, message: str) -> None:
        self.log.append(message)
