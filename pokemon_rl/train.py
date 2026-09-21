"""Self-play Deep Q-learning training loop with logging."""

from __future__ import annotations

import argparse
import csv
import logging
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

from .agents import DQNAgent, FrozenPolicy, GreedyAgent, RandomAgent
from .selfplay import Policy, play_episode

LOGGER = logging.getLogger("pokemon_rl")


@dataclass
class EvalResult:
    wins: int
    losses: int
    draws: int

    @property
    def battles(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        return self.wins / self.battles if self.battles else 0.0

    def __str__(self) -> str:
        return f"{self.win_rate:.1%} ({self.wins}W/{self.losses}L/{self.draws}D)"


def make_opponent(kind: str, seed: int) -> Policy:
    if kind == "random":
        return RandomAgent(seed=seed)
    if kind == "greedy":
        return GreedyAgent(seed=seed)
    raise ValueError(f"unknown opponent: {kind}")


def evaluate(agent: DQNAgent, opponent_kind: str, battles: int, seed: int,
             max_turns: int) -> EvalResult:
    """Greedy-policy evaluation; the agent plays each side equally often."""
    rng = random.Random(seed)
    result = EvalResult(0, 0, 0)
    for i in range(battles):
        opponent = make_opponent(opponent_kind, seed=seed + i)
        agent_side = i % 2
        policies: list[Policy] = [agent, agent]
        policies[1 - agent_side] = opponent
        episode = play_episode(policies, rng, greedy=True, max_turns=max_turns)
        if episode.winner is None:
            result.draws += 1
        elif episode.winner == agent_side:
            result.wins += 1
        else:
            result.losses += 1
    return result


def setup_logging(log_dir: Path, verbose: bool) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.handlers.clear()

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(console)

    file_handler = logging.FileHandler(log_dir / "train.log", mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(file_handler)


def train(args: argparse.Namespace) -> DQNAgent:
    log_dir = Path(args.log_dir)
    setup_logging(log_dir, args.verbose)
    rng = random.Random(args.seed)

    agent = DQNAgent(
        hidden_layer_sizes=tuple(int(h) for h in args.hidden.split(",")),
        learning_rate=args.lr,
        gamma=args.gamma,
        buffer_size=args.buffer,
        batch_size=args.batch,
        warmup=args.warmup,
        target_sync=args.target_sync,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay,
        seed=args.seed,
    )

    metrics_path = log_dir / "metrics.csv"
    metrics_file = metrics_path.open("w", newline="")
    metrics = csv.writer(metrics_file)
    metrics.writerow(["episode", "train_steps", "epsilon", "loss",
                      "avg_turns", "vs_random", "vs_greedy"])

    LOGGER.info("training self-play DQN for %d episodes (seed=%d)", args.episodes, args.seed)
    recent_turns: list[int] = []
    recent_losses: list[float] = []
    pool: list[FrozenPolicy] = []
    best_score = -1.0
    best_path = Path(args.out).with_name(Path(args.out).stem + "-best.pkl")

    for episode in range(1, args.episodes + 1):
        losses: list[float] = []
        policies: list[Policy]
        learner_sides: tuple[int, ...]

        def on_transition() -> None:
            for _ in range(args.updates_per_transition):
                loss = agent.train_step()
                if loss is not None:
                    losses.append(loss)

        if pool and rng.random() < args.pool_prob:
            # Most games (70% by default) are played against a league member
            # rather than the live policy, with the learner alternating sides.
            opponent = rng.choice(pool)
            learner_side = rng.randrange(2)
            policies = [agent, agent]
            policies[1 - learner_side] = opponent
            learner_sides = (learner_side,)
        else:
            policies, learner_sides = [agent, agent], (0, 1)

        result = play_episode(
            policies,
            rng,
            learner=agent,
            learner_sides=learner_sides,
            shaping=args.shaping,
            max_turns=args.max_turns,
            on_transition=on_transition,
        )
        recent_turns.append(result.turns)
        recent_losses.extend(losses)

        if args.snapshot_every and episode % args.snapshot_every == 0:
            snapshot = agent.snapshot(epsilon=args.epsilon_end, seed=args.seed + episode)
            if len(pool) < args.pool_size:
                pool.append(snapshot)
            else:
                # Replace a random member rather than the oldest, so the league
                # keeps early strategies around instead of drifting with the
                # learner -- the fix for chasing itself around the triangle.
                pool[rng.randrange(len(pool))] = snapshot
            LOGGER.debug("episode %5d | snapshot added, pool size %d", episode, len(pool))

        if args.transcript_every and episode % args.transcript_every == 0:
            LOGGER.debug("battle transcript, episode %d:\n  %s",
                         episode, "\n  ".join(result.log))

        if episode % args.report_every == 0:
            LOGGER.info(
                "episode %5d | steps %6d | eps %.3f | loss %8.4f | turns %.1f | buffer %d",
                episode, agent.train_steps, agent.epsilon,
                statistics.fmean(recent_losses) if recent_losses else float("nan"),
                statistics.fmean(recent_turns), len(agent.buffer),
            )

        if episode % args.eval_every == 0 or episode == args.episodes:
            vs_random = evaluate(agent, "random", args.eval_battles, args.seed + episode, args.max_turns)
            vs_greedy = evaluate(agent, "greedy", args.eval_battles, args.seed + episode, args.max_turns)
            LOGGER.info("episode %5d | eval vs random %s | vs greedy %s",
                        episode, vs_random, vs_greedy)
            score = (vs_random.win_rate + vs_greedy.win_rate) / 2
            if score > best_score:
                best_score = score
                best_path.parent.mkdir(parents=True, exist_ok=True)
                agent.save(best_path)
                LOGGER.info("episode %5d | new best (%.1f%% mean win rate), saved to %s",
                            episode, 100 * score, best_path)
            metrics.writerow([
                episode, agent.train_steps, f"{agent.epsilon:.4f}",
                f"{statistics.fmean(recent_losses):.6f}" if recent_losses else "",
                f"{statistics.fmean(recent_turns):.2f}",
                f"{vs_random.win_rate:.4f}", f"{vs_greedy.win_rate:.4f}",
            ])
            metrics_file.flush()
            recent_turns.clear()
            recent_losses.clear()

    metrics_file.close()
    model_path = Path(args.out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(model_path)
    LOGGER.info("saved model to %s, metrics to %s", model_path, metrics_path)

    sample = play_episode([agent, RandomAgent(seed=args.seed)], rng,
                          greedy=True, max_turns=args.max_turns)
    LOGGER.info("sample battle (trained P1 vs random P2):\n  %s", "\n  ".join(sample.log))
    return agent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Self-play DQN for a simplified Gen-1 battle")
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", default="128,128", help="comma separated layer sizes")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--buffer", type=int, default=50_000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1_000)
    parser.add_argument("--target-sync", type=int, default=1_000)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=int, default=30_000)
    parser.add_argument("--updates-per-transition", type=int, default=1)
    parser.add_argument("--pool-size", type=int, default=20,
                        help="how many past snapshots to keep as opponents")
    parser.add_argument("--snapshot-every", type=int, default=100,
                        help="add a snapshot to the opponent pool this often (0 disables)")
    parser.add_argument("--pool-prob", type=float, default=0.7,
                        help="probability of playing a pooled snapshot instead of pure self-play")
    parser.add_argument("--shaping", type=float, default=0.5,
                        help="weight on the HP-differential reward (0 = win/loss only)")
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-battles", type=int, default=200)
    parser.add_argument("--report-every", type=int, default=50)
    parser.add_argument("--transcript-every", type=int, default=500,
                        help="log a full battle transcript this often (0 disables)")
    parser.add_argument("--log-dir", default="runs")
    parser.add_argument("--out", default="runs/dqn.pkl")
    parser.add_argument("--verbose", action="store_true", help="print transcripts to console")
    return parser


def main(argv: list[str] | None = None) -> None:
    train(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
