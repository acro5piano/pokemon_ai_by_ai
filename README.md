# Pokemon self-play DQN

Deep Q-learning on a stripped-down Generation-1 battle, trained purely by
self-play. Everything is implemented from scratch: the battle engine, the
replay buffer and the Q-learning loop. The only ML dependency is
scikit-learn's `MLPRegressor`, and there is no RL framework involved.

## Rules

* Generation 1 mechanics, level 100, maximum DVs / stat experience.
* Every move hits (100% accuracy), no critical hits, no status moves, no
  secondary effects, no PP limits.
* Both trainers use the same fixed three Pokemon; only the order of play
  (the lead, and every later switch) is decided by the agent.

| Pokemon | Type | HP / Atk / Def / Spc / Spe | Moves |
| --- | --- | --- | --- |
| Rhydon | Ground / Rock | 413 / 358 / 338 / 188 / 178 | Earthquake, Rock Slide |
| Starmie | Water / Psychic | 323 / 248 / 268 / 298 / 328 | Surf, Blizzard |
| Zapdos | Electric / Flying | 383 / 278 / 268 / 348 / 298 | Drill Peck, Thunderbolt |

The matchups are a rock-paper-scissors triangle: Starmie's Surf is 4x on
Rhydon, Rhydon's Rock Slide is 2x on Zapdos, Zapdos' Thunderbolt is 2x on
Starmie — and Earthquake/Thunderbolt are outright immune against Zapdos/Rhydon.
Damage keeps the Gen-1 random roll (217..255), so battles stay stochastic.

## Battle model

* **Actions (5)** — `move 0`, `move 1`, `switch to slot 0/1/2`. Illegal actions
  (switching to the active or a fainted Pokemon, moving during a forced
  switch) are masked out both when acting and when bootstrapping targets.
* **Phases** — `lead` (both trainers choose a lead), `move` (simultaneous
  selection: switches resolve first, then moves in speed order), `replace`
  (a fainted side sends in a replacement). The battle is a draw if it hits
  the turn limit.
* **Observation (19 floats)** — for each of the six Pokemon, own team first:
  HP fraction, alive flag and active flag; plus a flag for whether the side is
  choosing a replacement rather than a move.
* **Reward** — `+1` win, `-1` loss, `0` draw, plus `--shaping` times the change
  in the HP differential (own remaining team HP minus the opponent's, as a
  fraction) accumulated since the side's previous decision.

## Learning

`DQNAgent` (`pokemon_rl/agents.py`) is a standard DQN:

* `MLPRegressor(hidden_layer_sizes=(128, 128))` with five outputs — one
  Q-value per action — updated by a single `partial_fit` per replay batch,
  where only the taken action's target differs from the current prediction.
* Uniform replay buffer, linear epsilon decay, and a target network (a deep
  copy of the regressor) resynced every `--target-sync` updates.
* Self-play: one network plays both sides and stores transitions from each
  side's own perspective. A transition spans from a side's decision to its
  next decision, which may be several battle steps later; the intervening HP
  swing lands in the shaping reward.
* League: every `--snapshot-every` episodes the weights are frozen into the
  opponent pool, and `--pool-prob` of the games are played against a random
  pool member instead of the live policy (the learner then only learns from
  its own side). Once the pool is full a *random* member is replaced rather
  than the oldest, so early strategies stay in the league.

Two scripted baselines are used for evaluation only: `RandomAgent` (uniform
legal action) and `GreedyAgent` (always the hardest-hitting move, best type
matchup on a forced switch).

## Usage

```bash
uv sync
uv run python main.py                      # 4000 episodes with the defaults
uv run python main.py --episodes 10000 --shaping 0.5 --verbose
uv run python -m pokemon_rl.train --help   # all options
uv run pytest                              # 54 tests
```

Output lands in `--log-dir` (default `runs/`):

* `train.log` — progress lines, periodic evaluation against both baselines,
  and full battle transcripts every `--transcript-every` episodes;
* `metrics.csv` — episode, update count, epsilon, TD loss, average battle
  length and both win rates;
* `runs/dqn.pkl` — the pickled Q-network (`DQNAgent.load` restores it);
* `runs/dqn-best.pkl` — the checkpoint with the best mean evaluation win rate.

A progress line and an evaluation line look like this:

```
episode  2000 | steps  74522 | eps 0.050 | loss   0.0177 | turns 36.5 | buffer 50000
episode  2000 | eval vs random 96.0% (192W/8L/0D) | vs greedy 67.0% (134W/66L/0D)
```

## Results

With the defaults (4000 episodes, ~12 minutes on a laptop CPU, seed 0) the
agent reaches **95-100% against the random baseline**, and averages roughly
**75% against the greedy baseline** over the second half of training — up
from ~56% before the league was added. The best checkpoint of that run wins
**100% against greedy** (200/200) and **90.5% against random**.

The score against the greedy baseline keeps oscillating between evaluations,
and that is the game rather than the optimiser: the three Pokemon form a
matchup triangle with no pure-strategy equilibrium, so a self-play learner
cycles around it, and a policy tuned against its current self is more or less
exploitable by any *fixed* opponent depending on where in the cycle it is.
Widening the league (`--pool-size`, `--snapshot-every`, `--pool-prob`) damps
the cycling; `dqn-best.pkl` keeps the strongest checkpoint regardless.

Inspecting a trained policy is a good sanity check on what it picked up: it
leads Starmie, and Surfs an opposing Rhydon out of the game on turn one (4x,
a guaranteed OHKO). Weaker checkpoints give themselves away by answering an
opposing Zapdos with their own Zapdos rather than Rhydon, which is the clean
counter.

## Layout

```
pokemon_rl/data.py      types, moves, species, type chart
pokemon_rl/battle.py    battle engine: phases, damage, legal actions, observations
pokemon_rl/agents.py    DQN agent, replay buffer, random/greedy baselines
pokemon_rl/selfplay.py  episode runner shared by training and evaluation
pokemon_rl/train.py     self-play training loop, evaluation, logging, CLI
tests/                  pytest suite for the engine, the agent and the loop
```
