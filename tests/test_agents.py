import random
from typing import Any

import numpy as np
import pytest

from pokemon_rl.agents import DQNAgent, GreedyAgent, RandomAgent, ReplayBuffer, Transition
from pokemon_rl.battle import MOVE_ACTIONS, N_ACTIONS, OBS_SIZE, Battle


def make_agent(**kwargs) -> DQNAgent:
    defaults: dict[str, Any] = dict(hidden_layer_sizes=(16,), batch_size=4, warmup=4, seed=0)
    defaults.update(kwargs)
    return DQNAgent(**defaults)


def test_q_values_have_one_output_per_action():
    agent = make_agent()
    assert agent.q_values(np.zeros(OBS_SIZE)).shape == (N_ACTIONS,)


def test_act_only_returns_legal_actions():
    agent = make_agent()
    rng = random.Random(0)
    mask = np.array([False, True, False, True, False])
    for greedy in (False, True):
        for _ in range(50):
            assert mask[agent.act(rng.random() * np.ones(OBS_SIZE), mask, greedy=greedy)]


def test_greedy_action_follows_the_q_values():
    agent = make_agent(epsilon_start=0.0, epsilon_end=0.0)
    state = np.ones(OBS_SIZE)
    mask = np.array([True, True, False, False, False])
    q = agent.q_values(state)
    assert agent.act(state, mask, greedy=True) == int(np.argmax(q[:2]))


def test_epsilon_decays_to_its_floor():
    agent = make_agent(epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_steps=100)
    assert agent.epsilon == 1.0
    agent.train_steps = 50
    assert agent.epsilon == 0.55
    agent.train_steps = 1000
    assert agent.epsilon == pytest.approx(0.1)


def test_train_step_waits_for_the_warmup_then_learns():
    agent = make_agent(warmup=8, batch_size=4)
    mask = np.ones(N_ACTIONS, dtype=bool)
    assert agent.train_step() is None
    for i in range(8):
        agent.remember(Transition(np.zeros(OBS_SIZE), i % N_ACTIONS, 1.0,
                                  np.zeros(OBS_SIZE), mask, False))
    assert agent.train_step() is not None
    assert agent.train_steps == 1


def test_learning_moves_the_q_value_of_a_rewarded_action():
    agent = make_agent(warmup=1, batch_size=8, learning_rate=0.05)
    state = np.ones(OBS_SIZE)
    mask = np.zeros(N_ACTIONS, dtype=bool)
    for _ in range(8):
        agent.remember(Transition(state, 0, 10.0, state, mask, True))
    before = agent.q_values(state)[0]
    for _ in range(50):
        agent.train_step()
    assert agent.q_values(state)[0] > before


def test_terminal_transitions_ignore_the_next_state_value():
    agent = make_agent(warmup=1, batch_size=1, gamma=0.9)
    state = np.ones(OBS_SIZE)
    agent.remember(Transition(state, 0, 1.0, state, np.zeros(N_ACTIONS, dtype=bool), True))
    # A terminal target is exactly the reward, so the reported TD loss is
    # (Q(s,a) - reward)^2 rather than anything involving the next state.
    expected = (agent.q_values(state)[0] - 1.0) ** 2
    assert agent.train_step() == expected


def test_target_network_syncs_on_schedule():
    agent = make_agent(warmup=1, batch_size=2, target_sync=3)
    mask = np.ones(N_ACTIONS, dtype=bool)
    for _ in range(2):
        agent.remember(Transition(np.ones(OBS_SIZE), 1, 1.0, np.ones(OBS_SIZE), mask, False))
    state = np.ones(OBS_SIZE)
    for _ in range(2):
        agent.train_step()
    assert not np.allclose(agent.model.predict(state.reshape(1, -1)),
                           agent.target_model.predict(state.reshape(1, -1)))
    agent.train_step()  # third step triggers the sync
    assert np.allclose(agent.model.predict(state.reshape(1, -1)),
                       agent.target_model.predict(state.reshape(1, -1)))


def test_save_and_load_round_trip(tmp_path):
    agent = make_agent(warmup=1, batch_size=2)
    mask = np.ones(N_ACTIONS, dtype=bool)
    for _ in range(4):
        agent.remember(Transition(np.ones(OBS_SIZE), 2, 1.0, np.ones(OBS_SIZE), mask, False))
    for _ in range(5):
        agent.train_step()
    path = tmp_path / "agent.pkl"
    agent.save(path)

    restored = make_agent()
    restored.load(path)
    state = np.ones(OBS_SIZE)
    assert restored.train_steps == agent.train_steps
    assert np.allclose(restored.q_values(state), agent.q_values(state))


def test_replay_buffer_evicts_oldest():
    buffer = ReplayBuffer(2, random.Random(0))
    for i in range(3):
        buffer.add(Transition(np.zeros(OBS_SIZE), i, 0.0, np.zeros(OBS_SIZE),
                              np.ones(N_ACTIONS, dtype=bool), False))
    assert len(buffer) == 2
    assert [t.action for t in buffer.buffer] == [1, 2]


def greedy_for(battle: Battle, side: int) -> GreedyAgent:
    agent = GreedyAgent(seed=0)
    agent.attach(battle, side)
    return agent


def test_random_agent_stays_legal():
    agent = RandomAgent(seed=0)
    mask = np.array([False, False, True, False, True])
    assert all(mask[agent.act(np.zeros(OBS_SIZE), mask)] for _ in range(20))


def test_greedy_agent_picks_the_strongest_move():
    battle = Battle(rng=random.Random(0))
    battle.step({0: MOVE_ACTIONS + 1, 1: MOVE_ACTIONS + 0})  # Starmie vs Rhydon
    agent = greedy_for(battle, side=0)
    assert agent.act(battle.observation(0), battle.legal_mask(0)) == 0  # Surf, 4x


def test_greedy_agent_avoids_a_move_the_target_is_immune_to():
    battle = Battle(rng=random.Random(0))
    battle.step({0: MOVE_ACTIONS + 0, 1: MOVE_ACTIONS + 2})  # Rhydon vs Zapdos
    agent = greedy_for(battle, side=0)
    assert agent.act(battle.observation(0), battle.legal_mask(0)) == 1  # Rock Slide, not Earthquake


def test_greedy_agent_replaces_a_faint_with_the_best_matchup():
    battle = Battle(rng=random.Random(1))
    battle.step({0: MOVE_ACTIONS + 1, 1: MOVE_ACTIONS + 0})  # Starmie vs Rhydon
    battle.step({0: 0, 1: 0})  # Surf knocks Rhydon out
    agent = greedy_for(battle, side=1)
    # Against Starmie, Zapdos (Thunderbolt, 2x) beats sending in the other Starmie.
    assert agent.act(battle.observation(1), battle.legal_mask(1)) == MOVE_ACTIONS + 2


def test_snapshot_is_frozen_at_the_time_it_is_taken():
    agent = make_agent(warmup=1, batch_size=4, learning_rate=0.05)
    state = np.ones(OBS_SIZE)
    mask = np.ones(N_ACTIONS, dtype=bool)
    for _ in range(4):
        agent.remember(Transition(state, 0, 10.0, state, mask, True))
    snapshot = agent.snapshot(epsilon=0.0, seed=0)
    before = snapshot.model.predict(state.reshape(1, -1))
    for _ in range(50):
        agent.train_step()
    assert np.allclose(snapshot.model.predict(state.reshape(1, -1)), before)
    assert not np.allclose(agent.q_values(state), before[0])


def test_snapshot_acts_greedily_within_the_mask():
    agent = make_agent()
    snapshot = agent.snapshot(epsilon=0.0, seed=0)
    state = np.ones(OBS_SIZE)
    mask = np.array([False, True, True, False, False])
    q = snapshot.model.predict(state.reshape(1, -1))[0]
    assert snapshot.act(state, mask) == 1 + int(np.argmax(q[1:3]))


def test_snapshot_exploration_stays_legal():
    agent = make_agent()
    snapshot = agent.snapshot(epsilon=1.0, seed=0)
    mask = np.array([True, False, False, True, False])
    assert all(mask[snapshot.act(np.zeros(OBS_SIZE), mask)] for _ in range(20))
