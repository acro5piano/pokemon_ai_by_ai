"""Which of the three draftable teams is strongest?

Drafting 2 of 3 leaves exactly three teams, so the question is a 3x3 symmetric
zero-sum game.  This module fills in that payoff matrix by replaying every
pairing with the draft pinned (`Battle(bans=...)`) and a fixed policy on both
sides, then solves the matrix for its Nash equilibrium.

A single "strongest team" only exists if the matrix has a dominant row.  If the
three teams beat each other in a cycle there is no such row, and the honest
answer is the equilibrium mixture rather than one name.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from .agents import DQNAgent, GreedyAgent, RandomAgent
from .battle import Battle, MOVE_ACTIONS, PHASE_DRAFT
from .data import SPECIES, TEAM, TEAM_SIZE
from .selfplay import play_episode

# A team is identified by the roster slot it leaves out.
BANS: tuple[int, ...] = tuple(range(TEAM_SIZE))


def team_label(ban: int) -> str:
    return "/".join(SPECIES[name].name for slot, name in enumerate(TEAM) if slot != ban)


@dataclass
class MatchupResult:
    score: float  # win rate of the row team, draws counted as a half
    battles: int
    draws: int
    avg_turns: float


def play_matchup(make_policies, ban_row: int, ban_col: int, battles: int,
                 seed: int, max_turns: int, greedy: bool = True) -> MatchupResult:
    """Replay one pairing, alternating which side fields the row team.

    Alternating matters because the engine breaks speed ties with a coin flip
    and both sides choose simultaneously; playing only one seat would fold any
    residual seat advantage into the team's score.
    """
    rng = random.Random(seed)
    total = 0.0
    draws = 0
    turns = 0
    for i in range(battles):
        row_side = i % 2
        bans = [0, 0]
        bans[row_side] = ban_row
        bans[1 - row_side] = ban_col
        episode = play_episode(make_policies(seed + i), rng, greedy=greedy,
                               max_turns=max_turns, bans=bans)
        if episode.winner is None:
            total += 0.5
            draws += 1
        elif episode.winner == row_side:
            total += 1.0
        turns += episode.turns
    return MatchupResult(total / battles, battles, draws, turns / battles)


def payoff_matrix(make_policies, battles: int, seed: int, max_turns: int,
                  greedy: bool = True) -> tuple[np.ndarray, list[list[MatchupResult]]]:
    """Win rate of every team against every team, as a 3x3 matrix."""
    results: list[list[MatchupResult]] = []
    matrix = np.zeros((TEAM_SIZE, TEAM_SIZE))
    for row in BANS:
        row_results: list[MatchupResult] = []
        for col in BANS:
            result = play_matchup(make_policies, row, col, battles,
                                  seed + 100 * row + col, max_turns, greedy=greedy)
            row_results.append(result)
            matrix[row, col] = result.score
        results.append(row_results)
    return matrix, results


def symmetrise(matrix: np.ndarray) -> np.ndarray:
    """Average every cell with its mirror.

    Both seats draft from the same roster and play under the same rules, so the
    true matrix is antisymmetric (`M[i][j] + M[j][i] == 1`) with an even
    diagonal.  The two cells are sampled from independent battles, so averaging
    them enforces that by construction and halves the sampling variance.
    """
    return (matrix + (1.0 - matrix.T)) / 2.0


def standard_error(battles: int) -> float:
    """Sampling error of one symmetrised cell, at the worst case p = 0.5."""
    return (0.25 / (2 * battles)) ** 0.5


def nash_equilibrium(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """Optimal mixed draft for a symmetric zero-sum game, plus its game value.

    Solves `max v s.t. p @ A >= v, sum(p) == 1, p >= 0` where `A` is the win
    rate re-centred on 0 so that an even matchup scores zero.
    """
    payoff = matrix - 0.5
    n = len(payoff)
    # Variables: [p_0 .. p_{n-1}, v];  minimise -v.
    c = np.zeros(n + 1)
    c[-1] = -1.0
    a_ub = np.hstack([-payoff.T, np.ones((n, 1))])
    b_ub = np.zeros(n)
    a_eq = np.zeros((1, n + 1))
    a_eq[0, :n] = 1.0
    bounds = [(0.0, 1.0)] * n + [(None, None)]
    solution = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=[1.0], bounds=bounds)
    if not solution.success:
        raise RuntimeError(f"could not solve the draft game: {solution.message}")
    return np.asarray(solution.x[:n]), float(solution.x[-1])


def draft_preference(agent: DQNAgent) -> tuple[np.ndarray, int]:
    """What the trained network itself thinks of each ban, before any battle."""
    battle = Battle(rng=random.Random(0))
    assert battle.phase == PHASE_DRAFT
    q_values = agent.q_values(battle.observation(0))[MOVE_ACTIONS:]
    return q_values, int(np.argmax(q_values))


# ------------------------------------------------------------------ reporting


def format_matrix(matrix: np.ndarray, results: list[list[MatchupResult]]) -> str:
    width = max(len(team_label(ban)) for ban in BANS)
    header = " " * (width + 3) + "  ".join(f"{team_label(ban):>{width}}" for ban in BANS)
    lines = [header, " " * (width + 3) + "-" * (len(header) - width - 3)]
    for row in BANS:
        cells = "  ".join(f"{matrix[row, col]:>{width}.1%}" for col in BANS)
        lines.append(f"{team_label(row):>{width}} | {cells}")
    lines.append("")
    lines.append(f"{'average':>{width}} | " + "  ".join(
        f"{matrix[row].mean():>{width}.1%}" for row in BANS))

    draw_rate = np.array([[results[r][c].draws / results[r][c].battles for c in BANS]
                          for r in BANS])
    if draw_rate.any():
        lines.append("")
        lines.append("draw rate (a stalled matchup says nothing about the teams):")
        for row in BANS:
            cells = "  ".join(f"{draw_rate[row, col]:>{width}.1%}" for col in BANS)
            lines.append(f"{team_label(row):>{width}} | {cells}")

    turns = np.mean([[results[r][c].avg_turns for c in BANS] for r in BANS])
    lines.append("")
    lines.append(f"(rows are the team drafted; {results[0][0].battles} battles per cell, "
                 f"{turns:.1f} turns on average)")
    return "\n".join(lines)


def describe_pairings(matrix: np.ndarray, battles: int) -> str:
    """Read the beats-relation off the matrix, ignoring cells that are a tie.

    A pairing counts as decided only if it clears two standard errors; calling
    a 50.2% cell a win would invent structure that the samples do not support.
    """
    margin = 2 * standard_error(battles)
    lines = []
    beats: dict[int, set[int]] = {ban: set() for ban in BANS}
    for row in BANS:
        for col in BANS:
            if row >= col:
                continue
            score = matrix[row, col]
            if abs(score - 0.5) <= margin:
                lines.append(f"{team_label(row)} = {team_label(col)} "
                             f"({score:.1%}, within noise)")
            elif score > 0.5:
                beats[row].add(col)
                lines.append(f"{team_label(row)} > {team_label(col)} ({score:.1%})")
            else:
                beats[col].add(row)
                lines.append(f"{team_label(col)} > {team_label(row)} ({1 - score:.1%})")

    beaten_by = {ban: {other for other in BANS if ban in beats[other]} for ban in BANS}
    others = len(BANS) - 1
    dominant = [ban for ban in BANS if len(beats[ban]) == others]
    cyclic = all(len(beats[ban]) == 1 and len(beaten_by[ban]) == 1 for ban in BANS)
    if dominant:
        verdict = f"dominant team: {team_label(dominant[0])} -- it beats both of the others"
    elif cyclic:
        verdict = "no dominant team -- the three beat each other in a cycle"
    else:
        best = max(BANS, key=lambda ban: (len(beats[ban]), -len(beaten_by[ban])))
        verdict = (f"no strict dominance; best record is {team_label(best)} at "
                   f"{len(beats[best])}W-{len(beaten_by[best])}L-"
                   f"{others - len(beats[best]) - len(beaten_by[best])}D of {others}")
    return "\n".join(["  " + line for line in lines] + ["", "  " + verdict])


def report(title: str, matrix: np.ndarray, results: list[list[MatchupResult]]) -> None:
    battles = results[0][0].battles
    print(f"\n=== {title} ===\n")
    print(format_matrix(matrix, results))
    balanced = symmetrise(matrix)
    print(f"\n  (+/- {2 * standard_error(battles):.1%} at two standard errors, "
          f"after averaging each cell with its mirror)\n")
    print(describe_pairings(balanced, battles))
    mixture, value = nash_equilibrium(balanced)
    print("\n  optimal draft (Nash equilibrium):")
    for ban in BANS:
        print(f"    {team_label(ban):>16}  {mixture[ban]:6.1%}")
    print(f"  game value {value + 0.5:.1%} (50% means a fair game, as it must be "
          f"when both sides draft optimally)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Head-to-head payoff matrix over the three draftable teams")
    parser.add_argument("--model", default="runs/dqn-best.pkl")
    parser.add_argument("--hidden", default="128,128")
    parser.add_argument("--battles", type=int, default=400,
                        help="battles per cell of the 3x3 matrix")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--epsilon", type=float, default=0.0,
                        help="play the matrix with this much exploration instead of "
                             "fully greedily; a deterministic policy can lock into a "
                             "mutual-immunity stall that a little noise breaks")
    parser.add_argument("--baselines", action="store_true",
                        help="also run the matrix under the scripted policies")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    agent = DQNAgent(hidden_layer_sizes=tuple(int(h) for h in args.hidden.split(",")),
                     seed=args.seed)
    agent.load(args.model)

    q_values, preferred = draft_preference(agent)
    print(f"loaded {args.model} ({agent.train_steps} training steps)")
    print("\nthe network's own view of the draft (Q value per ban):")
    for ban in BANS:
        marker = "  <- its greedy pick" if ban == preferred else ""
        print(f"  ban {SPECIES[TEAM[ban]].name:<8} -> {team_label(ban):<16} "
              f"Q = {q_values[ban]:+.3f}{marker}")

    if args.epsilon > 0:
        # A frozen copy per seat, so the exploration that breaks a deadlock is
        # drawn independently on each side.
        make_policies = lambda seed: [agent.snapshot(epsilon=args.epsilon, seed=seed),
                                      agent.snapshot(epsilon=args.epsilon, seed=seed + 7)]
        title = f"teams played out by the trained DQN (epsilon {args.epsilon})"
    else:
        make_policies = lambda seed: [agent, agent]
        title = "teams played out by the trained DQN (greedy)"

    matrix, results = payoff_matrix(make_policies, args.battles, args.seed,
                                    args.max_turns, greedy=args.epsilon == 0.0)
    report(title, matrix, results)

    if args.baselines:
        for name, factory in (
            ("greedy", lambda seed: [GreedyAgent(seed=seed), GreedyAgent(seed=seed + 7)]),
            ("random", lambda seed: [RandomAgent(seed=seed), RandomAgent(seed=seed + 7)]),
        ):
            baseline, baseline_results = payoff_matrix(
                factory, args.battles, args.seed, args.max_turns)
            report(f"teams played out by the {name} baseline", baseline, baseline_results)


if __name__ == "__main__":
    main()
