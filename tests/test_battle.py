import random

import numpy as np
import pytest

from pokemon_rl.battle import (
    MOVE_ACTIONS, N_ACTIONS, OBS_SIZE, PHASE_DRAFT, PHASE_LEAD, PHASE_MOVE,
    PHASE_REPLACE, Battle, PokemonState, compute_damage,
)
from pokemon_rl.data import MOVES, PICK_SIZE, SPECIES, TEAM_SIZE, type_effectiveness


def mon(name):
    return PokemonState(SPECIES[name])


def drafted(seed=0, bans=(2, 2), max_turns=200):
    """A battle whose draft is already settled, so tests can start at the lead.

    `bans=(2, 2)` leaves both sides with Rhydon and Starmie.
    """
    return Battle(rng=random.Random(seed), max_turns=max_turns, bans=bans)


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


# ------------------------------------------------------------------- the draft


def test_battle_opens_with_a_draft_over_the_whole_roster():
    battle = Battle(rng=random.Random(0))
    assert battle.phase == PHASE_DRAFT
    # Picking 2 of 3 is expressed as banning 1 of 3, so every slot is legal.
    assert battle.legal_actions(0) == [MOVE_ACTIONS + slot for slot in range(TEAM_SIZE)]
    assert battle.needs_action(0) and battle.needs_action(1)


def test_draft_leaves_each_side_with_two_pokemon():
    battle = Battle(rng=random.Random(0))
    battle.step({0: MOVE_ACTIONS + 2, 1: MOVE_ACTIONS + 0})
    assert battle.phase == PHASE_LEAD
    assert battle.team_names(0) == ("Rhydon", "Starmie")
    assert battle.team_names(1) == ("Starmie", "Zapdos")
    assert len(battle.picked_slots(0)) == PICK_SIZE
    assert battle.alive_slots(0) == [0, 1] and battle.alive_slots(1) == [1, 2]


def test_banned_pokemon_cannot_be_led_with_or_switched_to():
    battle = Battle(rng=random.Random(0))
    battle.step({0: MOVE_ACTIONS + 2, 1: MOVE_ACTIONS + 2})  # both drop Zapdos
    assert battle.legal_actions(0) == [MOVE_ACTIONS + 0, MOVE_ACTIONS + 1]
    battle.step({0: MOVE_ACTIONS + 0, 1: MOVE_ACTIONS + 0})
    assert battle.legal_actions(0) == [0, 1, MOVE_ACTIONS + 1]  # never slot 2


def test_bans_argument_skips_the_draft_phase():
    battle = drafted(bans=(0, 1))
    assert battle.phase == PHASE_LEAD
    assert battle.team_names(0) == ("Starmie", "Zapdos")
    assert battle.team_names(1) == ("Rhydon", "Zapdos")


def test_a_ban_outside_the_roster_is_rejected():
    with pytest.raises(ValueError):
        Battle(rng=random.Random(0), bans=(TEAM_SIZE, 0))


def test_a_move_action_is_illegal_during_the_draft():
    battle = Battle(rng=random.Random(0))
    with pytest.raises(ValueError):
        battle.step({0: 0, 1: MOVE_ACTIONS + 2})


# ------------------------------------------------------------------ the battle


def test_lead_phase_only_allows_switches():
    battle = drafted()
    assert battle.legal_actions(0) == [MOVE_ACTIONS + 0, MOVE_ACTIONS + 1]
    battle.step({0: MOVE_ACTIONS + 0, 1: MOVE_ACTIONS + 1})
    assert battle.phase == PHASE_MOVE
    assert battle.require_active(0).species.name == "Rhydon"
    assert battle.require_active(1).species.name == "Starmie"


def test_move_phase_excludes_switch_to_active_pokemon():
    battle = drafted()
    battle.step({0: MOVE_ACTIONS + 0, 1: MOVE_ACTIONS + 0})
    assert battle.legal_actions(0) == [0, 1, MOVE_ACTIONS + 1]


def test_illegal_action_is_rejected():
    battle = drafted()
    with pytest.raises(ValueError):
        battle.step({0: 0, 1: MOVE_ACTIONS + 0})  # no move actions during the lead phase


def test_faster_pokemon_moves_first():
    battle = drafted()
    battle.step({0: MOVE_ACTIONS + 0, 1: MOVE_ACTIONS + 1})  # Rhydon vs Starmie
    battle.step({0: 0, 1: 0})
    used = [line for line in battle.log if "used" in line]
    assert used[0].startswith("P2 Starmie")  # speed 328 > 178


def test_fainting_triggers_a_replacement_phase():
    battle = drafted(seed=1, bans=(0, 1))  # P1: Starmie/Zapdos, P2: Rhydon/Zapdos
    battle.step({0: MOVE_ACTIONS + 1, 1: MOVE_ACTIONS + 0})  # Starmie vs Rhydon
    battle.step({0: 0, 1: 0})  # Surf is a 4x OHKO on Rhydon
    assert battle.team[1][0].fainted
    assert battle.phase == PHASE_REPLACE
    assert not battle.needs_action(0)
    assert battle.needs_action(1)
    # With two drafted Pokemon the replacement is forced: only Zapdos is left.
    assert battle.legal_actions(1) == [MOVE_ACTIONS + 2]


def test_fainted_pokemon_does_not_attack():
    battle = drafted(seed=1, bans=(0, 1))
    battle.step({0: MOVE_ACTIONS + 1, 1: MOVE_ACTIONS + 0})  # Starmie (fast) vs Rhydon
    battle.step({0: 0, 1: 0})
    used = [line for line in battle.log if "used" in line]
    assert len(used) == 1 and used[0].startswith("P1 Starmie")


# ----------------------------------------------------------- the observation


def test_observation_shape_and_perspective():
    battle = drafted()
    battle.step({0: MOVE_ACTIONS + 0, 1: MOVE_ACTIONS + 1})
    obs0, obs1 = battle.observation(0), battle.observation(1)
    assert obs0.shape == (OBS_SIZE,)
    # Each side sees its own team in the first half.
    assert list(obs0[:12]) == list(obs1[12:24])
    assert obs0[3] == 1.0       # my slot 0 is active
    assert obs0[12 + 7] == 1.0  # their slot 1 is active


def test_observation_phase_flags_are_one_hot():
    battle = Battle(rng=random.Random(0))
    assert list(battle.observation(0)[-3:]) == [1.0, 0.0, 0.0]  # draft
    battle.step({0: MOVE_ACTIONS + 2, 1: MOVE_ACTIONS + 2})
    assert list(battle.observation(0)[-3:]) == [0.0, 1.0, 0.0]  # lead
    battle.step({0: MOVE_ACTIONS + 0, 1: MOVE_ACTIONS + 0})
    assert list(battle.observation(0)[-3:]) == [0.0, 0.0, 1.0]  # move


def test_observation_tells_a_banned_pokemon_apart_from_a_fainted_one():
    """The whole reason each slot carries `picked` as well as `alive`.

    Both a Pokemon that was never drafted and one that has been knocked out sit
    at 0 HP and are not alive; without the extra flag they would be identical,
    and a side could not tell how many knock-outs still separate it from a win.
    """
    battle = drafted(seed=1, bans=(0, 2))  # P2 drafts Rhydon and Starmie
    battle.step({0: MOVE_ACTIONS + 1, 1: MOVE_ACTIONS + 0})  # Starmie vs Rhydon
    battle.step({0: 0, 1: 0})  # Surf knocks Rhydon out
    battle.step({1: MOVE_ACTIONS + 1})  # P2 sends in its Starmie

    obs = battle.observation(1)
    fainted = list(obs[0:4])   # slot 0: drafted Rhydon, knocked out
    banned = list(obs[8:12])   # slot 2: Zapdos, never drafted
    assert fainted == [1.0, 0.0, 0.0, 0.0]
    assert banned == [0.0, 0.0, 0.0, 0.0]
    assert fainted != banned and fainted[1:] == banned[1:]


def test_legal_mask_matches_legal_actions():
    battle = drafted()
    battle.step({0: MOVE_ACTIONS + 0, 1: MOVE_ACTIONS + 0})
    mask = battle.legal_mask(0)
    assert mask.shape == (N_ACTIONS,)
    assert np.flatnonzero(mask).tolist() == battle.legal_actions(0)


# ------------------------------------------------------------------ the result


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
            loser = battle.team[1 - battle.winner]
            assert all(mon.fainted for mon in loser)
            assert any(not mon.fainted for mon in battle.team[battle.winner])


def test_turn_limit_ends_in_a_draw():
    battle = drafted(bans=(1, 1), max_turns=2)  # both sides: Rhydon and Zapdos
    battle.step({0: MOVE_ACTIONS + 2, 1: MOVE_ACTIONS + 0})  # Zapdos vs Rhydon
    for _ in range(2):
        battle.step({0: 1, 1: 0})  # Thunderbolt and Earthquake, both immune
    assert battle.done and battle.winner is None


def test_hp_diff_is_symmetric_and_normalised_to_the_drafted_team():
    battle = drafted(seed=1, bans=(0, 1))
    assert battle.hp_diff(0) == 0.0  # two full Pokemon each
    battle.step({0: MOVE_ACTIONS + 1, 1: MOVE_ACTIONS + 0})
    battle.step({0: 0, 1: 0})  # Surf knocks Rhydon out
    assert battle.hp_diff(0) == pytest.approx(-battle.hp_diff(1))
    assert battle.hp_diff(0) > 0
    # One of two Pokemon gone is half of this side's team, not a third of it.
    assert battle.hp_diff(0) == pytest.approx(0.5)
