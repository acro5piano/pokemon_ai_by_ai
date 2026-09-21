import random

import numpy as np
import pytest

from pokemon_rl.battle import (
    MOVE_ACTIONS, N_ACTIONS, OBS_SIZE, PHASE_MOVE, PHASE_REPLACE, Battle,
    PokemonState, compute_damage,
)
from pokemon_rl.data import MOVES, SPECIES, type_effectiveness


def mon(name):
    return PokemonState(SPECIES[name])


@pytest.mark.parametrize(
    "move,defender,expected",
    [
        ("surf", "rhydon", 4.0),        # water vs ground/rock
        ("surf", "starmie", 0.5),
        ("thunderbolt", "rhydon", 0.0),  # electric vs ground
        ("thunderbolt", "starmie", 2.0),
        ("thunderbolt", "zapdos", 1.0),  # 0.5 electric * 2 flying
        ("earthquake", "zapdos", 0.0),
        ("earthquake", "rhydon", 2.0),
        ("rock-slide", "rhydon", 0.5),
        ("rock-slide", "zapdos", 2.0),
        ("blizzard", "rhydon", 2.0),
        ("blizzard", "zapdos", 2.0),
        ("drill-peck", "zapdos", 0.5),
        ("drill-peck", "starmie", 1.0),
    ],
)
def test_type_effectiveness(move, defender, expected):
    assert type_effectiveness(MOVES[move].type, SPECIES[defender].types) == expected


def test_immune_moves_deal_no_damage():
    assert compute_damage(mon("zapdos"), mon("rhydon"), MOVES["thunderbolt"]) == 0
    assert compute_damage(mon("rhydon"), mon("zapdos"), MOVES["earthquake"]) == 0


def test_damage_roll_is_bounded_and_stab_helps():
    starmie, rhydon = mon("starmie"), mon("rhydon")
    low = compute_damage(starmie, rhydon, MOVES["surf"], roll=217)
    high = compute_damage(starmie, rhydon, MOVES["surf"], roll=255)
    assert 0 < low < high
    # Surf (STAB, 4x) hits Rhydon far harder than Blizzard (no STAB, 2x).
    assert high > compute_damage(starmie, rhydon, MOVES["blizzard"], roll=255)


def test_damage_never_rounds_down_to_zero():
    zapdos, rhydon = mon("zapdos"), mon("rhydon")
    assert compute_damage(zapdos, rhydon, MOVES["drill-peck"], roll=217) >= 1


def test_lead_phase_only_allows_switches():
    battle = Battle(rng=random.Random(0))
    assert battle.legal_actions(0) == [2, 3, 4]
    battle.step({0: 2, 1: 3})
    assert battle.phase == PHASE_MOVE
    assert battle.require_active(0).species.name == "Rhydon"
    assert battle.require_active(1).species.name == "Starmie"


def test_move_phase_excludes_switch_to_active_pokemon():
    battle = Battle(rng=random.Random(0))
    battle.step({0: 2, 1: 2})
    assert battle.legal_actions(0) == [0, 1, 3, 4]


def test_illegal_action_is_rejected():
    battle = Battle(rng=random.Random(0))
    with pytest.raises(ValueError):
        battle.step({0: 0, 1: 2})  # no move actions during the lead phase


def test_faster_pokemon_moves_first():
    battle = Battle(rng=random.Random(0))
    battle.step({0: MOVE_ACTIONS + 0, 1: MOVE_ACTIONS + 1})  # Rhydon vs Starmie
    battle.step({0: 0, 1: 0})
    used = [line for line in battle.log if "used" in line]
    assert used[0].startswith("P2 Starmie")  # speed 328 > 178


def test_fainting_triggers_a_replacement_phase():
    battle = Battle(rng=random.Random(1))
    battle.step({0: MOVE_ACTIONS + 1, 1: MOVE_ACTIONS + 0})  # Starmie vs Rhydon
    battle.step({0: 0, 1: 0})  # Surf is a 4x OHKO on Rhydon
    assert battle.team[1][0].fainted
    assert battle.phase == PHASE_REPLACE
    assert not battle.needs_action(0)
    assert battle.needs_action(1)
    assert battle.legal_actions(1) == [MOVE_ACTIONS + 1, MOVE_ACTIONS + 2]


def test_fainted_pokemon_does_not_attack():
    battle = Battle(rng=random.Random(1))
    battle.step({0: MOVE_ACTIONS + 1, 1: MOVE_ACTIONS + 0})  # Starmie (fast) vs Rhydon
    battle.step({0: 0, 1: 0})
    used = [line for line in battle.log if "used" in line]
    assert len(used) == 1 and used[0].startswith("P1 Starmie")


def test_observation_shape_and_perspective():
    battle = Battle(rng=random.Random(0))
    battle.step({0: 2, 1: 3})
    obs0, obs1 = battle.observation(0), battle.observation(1)
    assert obs0.shape == (OBS_SIZE,)
    # Each side sees its own team in the first half.
    assert list(obs0[:9]) == list(obs1[9:18])
    assert obs0[2] == 1.0 and obs0[9 + 5] == 1.0  # my slot 0 active, their slot 1


def test_legal_mask_matches_legal_actions():
    battle = Battle(rng=random.Random(0))
    battle.step({0: 2, 1: 2})
    mask = battle.legal_mask(0)
    assert mask.shape == (N_ACTIONS,)
    assert np.flatnonzero(mask).tolist() == battle.legal_actions(0)


def test_random_battles_terminate_with_a_verdict():
    rng = random.Random(7)
    for _ in range(30):
        battle = Battle(rng=rng, max_turns=200)
        while not battle.done:
            actions = {side: rng.choice(battle.legal_actions(side))
                       for side in (0, 1) if battle.needs_action(side)}
            battle.step(actions)
        assert battle.winner in (0, 1, None)
        if battle.winner is not None:
            assert all(m.fainted for m in battle.team[1 - battle.winner])
            assert any(not m.fainted for m in battle.team[battle.winner])


def test_turn_limit_ends_in_a_draw():
    rng = random.Random(0)
    battle = Battle(rng=rng, max_turns=2)
    battle.step({0: MOVE_ACTIONS + 2, 1: MOVE_ACTIONS + 0})  # Zapdos vs Rhydon
    for _ in range(2):
        battle.step({0: 1, 1: 0})  # Thunderbolt and Earthquake, both immune
    assert battle.done and battle.winner is None


def test_hp_diff_is_symmetric():
    battle = Battle(rng=random.Random(0))
    battle.step({0: MOVE_ACTIONS + 1, 1: MOVE_ACTIONS + 0})
    battle.step({0: 0, 1: 0})
    assert battle.hp_diff(0) == pytest.approx(-battle.hp_diff(1))
    assert battle.hp_diff(0) > 0
