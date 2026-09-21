# Pokemon self-play DQN

Deep Q-learning on a stripped-down Generation-1 battle, trained purely by
self-play. Everything is implemented from scratch: the battle engine, the
replay buffer and the Q-learning loop. The only ML dependency is
scikit-learn's `MLPRegressor`, and there is no RL framework involved.

## Rules

* Generation 1 mechanics, level 100, maximum DVs / stat experience.
* Every move hits (100% accuracy), no critical hits, no status moves, no
  secondary effects, no PP limits.
* Both trainers draft **two of the same fixed three Pokemon**, simultaneously
  and without seeing the other's choice. Picking 2 of 3 is the same decision
  as banning 1 of 3, so the draft is a single choice with three options.
* After the draft the agent decides the lead and every later switch.

| Pokemon | Type | HP / Atk / Def / Spc / Spe | Moves |
| --- | --- | --- | --- |
| Rhydon | Ground / Rock | 413 / 358 / 338 / 188 / 178 | Earthquake, Rock Slide |
| Starmie | Water / Psychic | 323 / 248 / 268 / 298 / 328 | Surf, Blizzard |
| Zapdos | Electric / Flying | 383 / 278 / 268 / 348 / 298 | Drill Peck, Thunderbolt |

The matchups are a rock-paper-scissors triangle: Starmie's Surf is 4x on
Rhydon, Rhydon's Rock Slide is 2x on Zapdos, Zapdos' Thunderbolt is 2x on
Starmie — and Earthquake/Thunderbolt are outright immune against Zapdos/Rhydon.
Damage keeps the Gen-1 random roll (217..255), so battles stay stochastic.

Drafting two of them leaves exactly three teams — Rhydon/Starmie,
Rhydon/Zapdos and Starmie/Zapdos — so "which team is strongest" is a 3x3
symmetric zero-sum game. `pokemon_rl/matchup.py` fills in that matrix and
solves it; see [Results](#results).

## Battle model

* **Actions (5)** — `move 0`, `move 1`, and one slot-addressed action per
  roster slot: a *ban* during the draft, a *switch* during the battle. Illegal
  actions (switching to the active, a fainted, or an undrafted Pokemon; moving
  during a forced switch) are masked out both when acting and when
  bootstrapping targets.
* **Phases** — `draft` (both trainers ban one of the three, simultaneously),
  `lead` (both choose a lead), `move` (simultaneous selection: switches resolve
  first, then moves in speed order), `replace` (a fainted side sends in its
  last Pokemon). The battle is a draw if it hits the turn limit.
* **Observation (27 floats)** — for each of the six roster slots, own roster
  first: drafted flag, HP fraction, alive flag, active flag; plus a one-hot of
  the three decision types (draft / lead-or-replace / move).

  The slot index *is* the species — the roster is fixed and ordered, so no
  species encoding is needed and one network plays both seats. `drafted` and
  `alive` are both carried because HP alone would collapse three distinct
  situations into one number: a Pokemon that was never drafted and one that has
  fainted both sit at 0 HP, and a side that cannot tell them apart cannot tell
  how many knock-outs still separate it from a win. The same reasoning is why
  `alive` exists at all rather than being read off `hp_fraction > 0`: one HP and
  zero HP differ by 0.003 in the input and by a whole Pokemon in the game.
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
* `--draft-epsilon` keeps an exploration floor on the draft alone. It is one
  decision per episode taken from an observation that is identical every game,
  so the decayed epsilon would try each alternative only a handful of times
  over a whole run; without the floor the learner locks onto one team and never
  becomes competent with the other two. The draft also earns no shaping reward
  of its own and is credited purely by bootstrapping from the battle that
  follows.
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
legal action) and `GreedyAgent` (always the hardest-hitting move, never
switching voluntarily). Both draft at random: with only three teams, every ban
is the right one against some opponent, so a fixed draft would just make the
baseline exploitable in one specific direction.

## Usage

```bash
uv sync
uv run python main.py --episodes 6000      # ~10 minutes on a laptop CPU
uv run python main.py --episodes 10000 --shaping 0.5 --verbose
uv run python -m pokemon_rl.train --help   # all options
uv run pytest                              # 76 tests

# Which team is strongest?  Fills in the 3x3 payoff matrix and solves it.
uv run python -m pokemon_rl.matchup --model runs/dqn-best.pkl --battles 400
uv run python -m pokemon_rl.matchup --baselines   # also under the scripted policies
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

6000 episodes, ~10 minutes on a laptop CPU, seed 0. The agent reaches
**95-97% against the random baseline** and swings between roughly **40% and
85% against the greedy baseline**; both baselines now draft at random, so a
fixed policy is exploitable in one direction or another depending on where in
the matchup cycle it currently sits.

### Which team is strongest?

`pokemon_rl/matchup.py` pins both teams (`Battle(bans=...)`), replays every
pairing, and solves the resulting 3x3 matrix. Five independent measurements:

| played by | strongest | weakest | Starmie/Zapdos vs Rhydon/Starmie |
| --- | --- | --- | --- |
| random baseline | Starmie/Zapdos | Rhydon/Zapdos | 70.0% |
| greedy baseline | Starmie/Zapdos | Rhydon/Zapdos | 75.4% |
| `dqn.pkl` (ep 6000) | Starmie/Zapdos | Rhydon/Zapdos | 55.6% |
| both checkpoints mixed | Starmie/Zapdos | Rhydon/Zapdos | 69.8% |
| `dqn-best.pkl` (ep 2250) | *Rhydon/Starmie* | Rhydon/Zapdos | 38.5% |

**Starmie/Zapdos is the strongest team**, and the equilibrium draft is to take
it outright — a pure strategy, not a mixture. Every measurement agrees that
**Rhydon/Zapdos is the worst**, by a wide margin (18.5% against Starmie/Zapdos
under mixed play).

The triangle that governs the three *Pokemon* does not survive drafting them in
*pairs*: a pair covers two corners of it at once, so the rock-paper-scissors
structure collapses and a dominant team appears. Starmie/Zapdos is the only
team with no dead matchup — every team containing Rhydon carries a Pokemon that
is 4x weak to Surf, and Rhydon/Zapdos is the pair whose two members are
mutually immune to each other's STAB, so it effectively plays a Pokemon down
against anything holding a Starmie.

Two caveats, both worth reading before trusting the table:

* **The measurement is policy-dependent.** `dqn-best.pkl` is the outlier
  because "best" means the best mean win rate against the two *scripted*
  baselines, not the strongest player; it happens to pilot Rhydon/Starmie
  better than it pilots Starmie/Zapdos. The two closest cells flip with the
  checkpoint, which is why the table reports several policies rather than one.
* **A fully greedy policy deadlocks.** Played at `--epsilon 0`, two of the nine
  cells are *100% draws over 200 turns*: Zapdos fires Thunderbolt at a Rhydon
  that is immune to it while that Rhydon fires Earthquake at a Zapdos that is
  immune to it, forever. Neither side is punished — a draw scores 0, which
  beats losing — so self-play never learns its way out of the fixed point.
  `--epsilon 0.05` breaks the cycle and drops the draw rate to ~0%.

### The agent does not know any of this

Its own Q values at the draft are nearly flat, and it picks the wrong team:

```
ban Rhydon   -> Starmie/Zapdos   Q = +0.611  <- its greedy pick
ban Starmie  -> Rhydon/Zapdos    Q = +0.599
ban Zapdos   -> Rhydon/Starmie   Q = +0.607
```

A spread of 0.012 across options whose true win rates differ by 40 points. This
is the credit-assignment problem the draft was always going to have: it is one
decision per episode, it earns no shaping reward, and the outcome reaches it
only after ~15 further decisions of discounting. Getting the draft learned
properly needs more than a Q head shared with the battle policy — a separate
head, a weaker discount on that one transition, or simply many more episodes.

Inspecting battle play is a better sanity check on what was learned: the agent
leads Starmie and Surfs an opposing Rhydon out of the game on turn one (4x, a
guaranteed OHKO).

## Layout

```
pokemon_rl/data.py      types, moves, species, type chart
pokemon_rl/battle.py    battle engine: phases, damage, legal actions, observations
pokemon_rl/agents.py    DQN agent, replay buffer, random/greedy baselines
pokemon_rl/selfplay.py  episode runner shared by training and evaluation
pokemon_rl/train.py     self-play training loop, evaluation, logging, CLI
pokemon_rl/matchup.py   team payoff matrix + Nash equilibrium of the draft
tests/                  pytest suite for the engine, the agent and the loop
```
