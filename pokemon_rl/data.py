"""Static game data: types, moves and species (Generation 1, level 100)."""

from __future__ import annotations

from dataclasses import dataclass

PHYSICAL = "physical"
SPECIAL = "special"

# Generation 1 type chart, restricted to the attacking types that appear in this
# game.  Only non-1.0 entries are listed; anything missing is neutral.
TYPE_CHART: dict[str, dict[str, float]] = {
    "ground": {"fire": 2.0, "electric": 2.0, "grass": 0.5, "poison": 2.0,
               "flying": 0.0, "bug": 0.5, "rock": 2.0},
    "rock": {"fire": 2.0, "ice": 2.0, "flying": 2.0, "bug": 2.0,
             "fighting": 0.5, "ground": 0.5},
    "water": {"fire": 2.0, "water": 0.5, "grass": 0.5, "ground": 2.0,
              "rock": 2.0, "dragon": 0.5},
    "ice": {"water": 0.5, "grass": 2.0, "ice": 0.5, "ground": 2.0,
            "flying": 2.0, "dragon": 2.0},
    "flying": {"electric": 0.5, "grass": 2.0, "fighting": 2.0, "bug": 2.0,
               "rock": 0.5},
    "electric": {"water": 2.0, "electric": 0.5, "grass": 0.5, "ground": 0.0,
                 "flying": 2.0, "dragon": 0.5},
}


def type_effectiveness(move_type: str, defender_types: tuple[str, ...]) -> float:
    """Combined damage multiplier of `move_type` against a defender."""
    row = TYPE_CHART[move_type]
    multiplier = 1.0
    for defender_type in defender_types:
        multiplier *= row.get(defender_type, 1.0)
    return multiplier


@dataclass(frozen=True)
class Move:
    name: str
    type: str
    power: int
    category: str  # PHYSICAL or SPECIAL


@dataclass(frozen=True)
class Species:
    name: str
    types: tuple[str, ...]
    hp: int
    attack: int
    defense: int
    special: int
    speed: int
    moves: tuple[Move, ...]


MOVES: dict[str, Move] = {
    "earthquake": Move("Earthquake", "ground", 100, PHYSICAL),
    "rock-slide": Move("Rock Slide", "rock", 75, PHYSICAL),
    "surf": Move("Surf", "water", 95, SPECIAL),
    "blizzard": Move("Blizzard", "ice", 120, SPECIAL),
    "drill-peck": Move("Drill Peck", "flying", 80, PHYSICAL),
    "thunderbolt": Move("Thunderbolt", "electric", 95, SPECIAL),
}

# Level 100 stats with maximum DVs / stat experience, the RBY competitive norm.
SPECIES: dict[str, Species] = {
    "rhydon": Species(
        name="Rhydon",
        types=("ground", "rock"),
        hp=413, attack=358, defense=338, special=188, speed=178,
        moves=(MOVES["earthquake"], MOVES["rock-slide"]),
    ),
    "starmie": Species(
        name="Starmie",
        types=("water", "psychic"),
        hp=323, attack=248, defense=268, special=298, speed=328,
        moves=(MOVES["surf"], MOVES["blizzard"]),
    ),
    "zapdos": Species(
        name="Zapdos",
        types=("electric", "flying"),
        hp=383, attack=278, defense=268, special=348, speed=298,
        moves=(MOVES["drill-peck"], MOVES["thunderbolt"]),
    ),
}

# Both trainers use this fixed roster; a team slot always maps to the same
# species, so the lead (and every later switch) is the only team decision.
TEAM: tuple[str, ...] = ("rhydon", "starmie", "zapdos")

LEVEL = 100
MOVES_PER_POKEMON = 2
TEAM_SIZE = len(TEAM)
