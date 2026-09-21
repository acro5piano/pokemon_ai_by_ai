import random

import numpy as np

from pokemon_rl.agents import DQNAgent, GreedyAgent, RandomAgent
from pokemon_rl.battle import N_ACTIONS, OBS_SIZE
from pokemon_rl.selfplay import play_episode
from pokemon_rl.train import build_parser, evaluate, train


def test_episode_between_random_agents_finishes():
    result = play_episode([RandomAgent(seed=1), RandomAgent(seed=2)], random.Random(3))
    assert result.winner in (0, 1, None)
    assert result.turns > 0 and result.decisions >= 2
    assert result.log[0].startswith("P1 leads with")


def test_self_play_collects_transitions_for_both_sides():
    agent = DQNAgent(hidden_layer_sizes=(8,), warmup=10_000, seed=0)
    play_episode([agent, agent], random.Random(0), learner=agent,
                 learner_sides=(0, 1), shaping=0.5)
    transitions = list(agent.buffer.buffer)
    assert len(transitions) >= 4
    for transition in transitions:
        assert transition.state.shape == (OBS_SIZE,)
        assert transition.next_mask.shape == (N_ACTIONS,)
        assert 0 <= transition.action < N_ACTIONS
    terminal = [t for t in transitions if t.done]
    assert len(terminal) == 2  # exactly one closing transition per side
    # Win and loss rewards mirror each other around the shaping term.
    assert not terminal[0].next_mask.any()
    assert any(abs(t.reward) > 0.5 for t in terminal)


def test_terminal_rewards_match_the_battle_outcome():
    agent = DQNAgent(hidden_layer_sizes=(8,), warmup=10_000, seed=0)
    for seed in range(5):
        agent.buffer.buffer.clear()
        result = play_episode([agent, agent], random.Random(seed), learner=agent,
                              learner_sides=(0, 1), shaping=0.0)
        terminal = [t for t in agent.buffer.buffer if t.done]
        rewards = [t.reward for t in terminal]
        if result.winner is None:
            assert rewards == [0.0, 0.0]
        else:
            assert sorted(rewards) == [-1.0, 1.0]


def test_shaping_reward_tracks_the_hp_swing():
    agent = DQNAgent(hidden_layer_sizes=(8,), warmup=10_000, seed=0)
    play_episode([agent, agent], random.Random(0), learner=agent,
                 learner_sides=(0, 1), shaping=1.0)
    non_terminal = [t for t in agent.buffer.buffer if not t.done]
    assert any(t.reward != 0.0 for t in non_terminal)
    assert all(abs(t.reward) <= 1.0 for t in non_terminal)


def test_learner_sides_filter_what_is_stored():
    agent = DQNAgent(hidden_layer_sizes=(8,), warmup=10_000, seed=0)
    play_episode([agent, RandomAgent(seed=1)], random.Random(0), learner=agent,
                 learner_sides=(0,), shaping=0.0)
    assert sum(1 for t in agent.buffer.buffer if t.done) == 1


def test_on_transition_fires_once_per_stored_transition():
    agent = DQNAgent(hidden_layer_sizes=(8,), warmup=10_000, seed=0)
    calls = []
    play_episode([agent, agent], random.Random(0), learner=agent, learner_sides=(0, 1),
                 on_transition=lambda: calls.append(1))
    assert len(calls) == len(agent.buffer.buffer)


def test_greedy_agent_is_reattached_to_each_new_battle():
    opponent = GreedyAgent(seed=0)
    battles = []
    for seed in range(3):
        result = play_episode([RandomAgent(seed=seed), opponent], random.Random(seed))
        assert result.winner in (0, 1, None)
        assert opponent.side == 1
        battles.append(id(opponent.battle))
    assert len(set(battles)) == 3


def test_evaluate_reports_a_complete_tally():
    agent = DQNAgent(hidden_layer_sizes=(8,), seed=0)
    result = evaluate(agent, "random", battles=6, seed=0, max_turns=200)
    assert result.battles == 6
    assert 0.0 <= result.win_rate <= 1.0


def test_training_run_writes_logs_and_a_model(tmp_path):
    args = build_parser().parse_args([
        "--episodes", "5", "--eval-every", "5", "--eval-battles", "4",
        "--report-every", "5", "--warmup", "10", "--hidden", "16",
        "--transcript-every", "5",
        "--log-dir", str(tmp_path), "--out", str(tmp_path / "dqn.pkl"),
    ])
    agent = train(args)
    assert agent.train_steps > 0
    assert (tmp_path / "dqn.pkl").exists()
    assert "vs_random" in (tmp_path / "metrics.csv").read_text()
    log = (tmp_path / "train.log").read_text()
    assert "eval vs random" in log and "battle transcript" in log

    reloaded = DQNAgent(hidden_layer_sizes=(16,), seed=0)
    reloaded.load(tmp_path / "dqn.pkl")
    assert np.allclose(reloaded.q_values(np.zeros(OBS_SIZE)), agent.q_values(np.zeros(OBS_SIZE)))


def test_pool_opponents_only_teach_the_learner_side(tmp_path):
    args = build_parser().parse_args([
        "--episodes", "6", "--eval-every", "6", "--eval-battles", "2",
        "--report-every", "6", "--warmup", "10", "--hidden", "16",
        "--snapshot-every", "1", "--pool-prob", "1.0",
        "--log-dir", str(tmp_path), "--out", str(tmp_path / "dqn.pkl"),
    ])
    agent = train(args)
    # From episode 2 on every game is against a snapshot, so most battles
    # contribute a single terminal transition instead of one per side.
    assert (tmp_path / "dqn-best.pkl").exists()
    assert "snapshot added" in (tmp_path / "train.log").read_text()
    assert agent.train_steps > 0
