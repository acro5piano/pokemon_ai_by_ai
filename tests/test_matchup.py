import numpy as np
import pytest

from pokemon_rl.agents import RandomAgent
from pokemon_rl.matchup import (
    BANS, describe_pairings, nash_equilibrium, payoff_matrix, play_matchup,
    standard_error, symmetrise, team_label,
)
from pokemon_rl.data import PICK_SIZE, TEAM_SIZE


def random_policies(seed):
    return [RandomAgent(seed=seed), RandomAgent(seed=seed + 7)]


def test_every_ban_names_a_distinct_team_of_the_drafted_size():
    labels = [team_label(ban) for ban in BANS]
    assert len(set(labels)) == TEAM_SIZE
    assert all(len(label.split("/")) == PICK_SIZE for label in labels)
    assert team_label(2) == "Rhydon/Starmie"


def test_a_matchup_scores_every_battle():
    result = play_matchup(random_policies, 0, 1, battles=8, seed=0, max_turns=200)
    assert result.battles == 8
    assert 0.0 <= result.score <= 1.0
    assert result.avg_turns > 0
    # Draws are half a point each, so the score lands on a multiple of 1/16.
    assert result.score * 16 == pytest.approx(round(result.score * 16))


def test_the_matrix_is_antisymmetric_within_sampling_noise():
    """A beats B exactly as often as B loses to A, so the two cells sum to 1.

    Each cell is sampled independently, so this is the check that catches a
    matchup scored from the wrong seat -- a mirrored bug would push the sums
    towards 0 or 2 rather than 1.
    """
    matrix, _ = payoff_matrix(random_policies, battles=60, seed=0, max_turns=200)
    assert matrix.shape == (TEAM_SIZE, TEAM_SIZE)
    for row in BANS:
        for col in BANS:
            if row < col:
                assert matrix[row, col] + matrix[col, row] == pytest.approx(1.0, abs=0.3)


def test_a_mirror_matchup_is_even():
    """Both seats field the same team and the same policy, so it is a coin flip."""
    result = play_matchup(random_policies, 1, 1, battles=40, seed=0, max_turns=200)
    assert result.score == pytest.approx(0.5, abs=0.25)


def test_nash_of_a_rock_paper_scissors_matrix_is_uniform():
    matrix = np.array([
        [0.5, 1.0, 0.0],
        [0.0, 0.5, 1.0],
        [1.0, 0.0, 0.5],
    ])
    mixture, value = nash_equilibrium(matrix)
    assert mixture == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=1e-6)
    assert value == pytest.approx(0.0, abs=1e-9)  # a fair game, re-centred on 0


def test_nash_of_a_dominant_matrix_is_pure():
    matrix = np.array([
        [0.5, 0.9, 0.8],
        [0.1, 0.5, 0.6],
        [0.2, 0.4, 0.5],
    ])
    mixture, value = nash_equilibrium(matrix)
    assert mixture == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)
    assert value == pytest.approx(0.0, abs=1e-9)


def test_nash_recovers_a_lopsided_mixture():
    """One team beaten by both others is dropped from the optimal draft."""
    matrix = np.array([
        [0.5, 0.6, 0.4],
        [0.4, 0.5, 0.7],
        [0.6, 0.3, 0.5],
    ])
    mixture, _ = nash_equilibrium(matrix)
    assert mixture.sum() == pytest.approx(1.0)
    assert (mixture >= -1e-9).all()


def test_symmetrising_enforces_an_antisymmetric_matrix():
    noisy = np.array([
        [0.52, 0.70, 0.10],
        [0.26, 0.48, 0.90],
        [0.94, 0.14, 0.51],
    ])
    balanced = symmetrise(noisy)
    assert np.allclose(np.diag(balanced), 0.5)
    assert np.allclose(balanced + balanced.T, 1.0)
    # Each cell is the average of its own sample and its mirror's complement.
    assert balanced[0, 1] == pytest.approx((0.70 + (1 - 0.26)) / 2)


def test_symmetrising_leaves_an_already_symmetric_matrix_alone():
    exact = np.array([
        [0.5, 0.8, 0.3],
        [0.2, 0.5, 0.6],
        [0.7, 0.4, 0.5],
    ])
    assert np.allclose(symmetrise(exact), exact)


def test_a_near_even_pairing_is_not_called_a_win():
    battles = 600
    edge = standard_error(battles)  # well inside the two-sigma margin
    matrix = symmetrise(np.array([
        [0.5, 0.5 + edge, 0.5 + edge],
        [0.5 - edge, 0.5, 0.5 + edge],
        [0.5 - edge, 0.5 - edge, 0.5],
    ]))
    text = describe_pairings(matrix, battles)
    assert text.count("within noise") == 3
    assert "no strict dominance" in text


def test_a_clear_winner_is_reported_as_dominant():
    matrix = symmetrise(np.array([
        [0.5, 0.8, 0.75],
        [0.2, 0.5, 0.65],
        [0.25, 0.35, 0.5],
    ]))
    text = describe_pairings(matrix, battles=600)
    assert "dominant team: Starmie/Zapdos" in text


def test_a_cycle_is_reported_as_a_cycle():
    matrix = symmetrise(np.array([
        [0.5, 0.8, 0.2],
        [0.2, 0.5, 0.8],
        [0.8, 0.2, 0.5],
    ]))
    text = describe_pairings(matrix, battles=600)
    assert "beat each other in a cycle" in text
