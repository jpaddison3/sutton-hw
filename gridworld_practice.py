"""Same as Frozen Lake from Berkeley Deep RL HW 2

Goal: Create and MDP, then solve.
"""
import itertools
from collections import Counter, defaultdict
import numpy as np
import nose.tools as nt

# = = = MDP Definition = = =
# TODO
grid = np.array([
    [0,      0, 0,  1],
    [0, np.nan, 0, -1],
    [0,      0, 0,  0],
])
# TODO Maybe remove these in favor of two space state
# TODO
# possible_states = np.arange(12)
possible_states = list(itertools.product(range(len(grid)), range(len(grid[0]))))
# TODO
state_to_grid = {state: coords for state, coords in
                 zip(possible_states,
                     itertools.product(range(len(grid)), range(len(grid[0]))))}
# TODO
grid_to_state = {coords: state for state, coords in
                 zip(possible_states,
                     itertools.product(range(len(grid)), range(len(grid[0]))))}
# TODO
actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]


def reward(state, action, next_state):
    """TODO

    returns: int, reward
    """
    if next_state == (1, 1):
        return 0
    if next_state[0] < 0 or len(grid) <= next_state[0] or \
       next_state[1] < 0 or len(grid[0]) <= next_state[1]:
        return 0
    return grid[next_state[0], next_state[1]]


# TODO Maybe not this either
def possible_next_states(state):
    """TODO

    """
    pass


def transition_probability(state, action, next_state):
    """TODO

    returns p in [0,1]
    """
    action = np.array(action)
    # print("grid", grid[state[0], state[1]])
    if np.allclose(state, [0, 3]) or np.allclose(state, [1, 3]):
        return 0
    if next_state[0] < 0 or len(grid) <= next_state[0] or \
       next_state[1] < 0 or len(grid[0]) <= next_state[1]:
        return 0
    if np.isnan(grid[next_state[0], next_state[1]]):
        return 0
    if np.allclose(next_state, state + action):
        return .8
    misaction = np.array([action[1], action[0]])
    if np.allclose(next_state, state + misaction):
        return .1
    misaction *= -1
    if np.allclose(next_state, state + misaction):
        return .1
    return 0


# = = = Solver = = =
GAMMA = .9
VALUE_ITERATIONS = 20
POLICY_ITERATIONS = 20


def compute_single_value(state, policy, previous_value):  # oh look
    """TODO

    """
    action = policy[state]
    # print("a", action)
    value = 0
    for next_state in possible_states:
        # Note that most next_states have prob = 0
        prob = transition_probability(state, action, next_state)
        got_reward = reward(state, action, next_state)
        # print("s', p, r, V(s'):", next_state, prob, got_reward, previous_value[next_state])
        value += prob * (got_reward + GAMMA * previous_value[next_state])
    return value


def value_from_policy(policy):
    """TODO

    returns map
    """
    value = {(state[0], state[1]): 0 for state in possible_states}
    for i in range(VALUE_ITERATIONS):
        value_prev = value.copy()
        for state in possible_states:
            value[state] = compute_single_value(state, policy, value_prev)
            # print("s, V_prev(s), V(s):", state, value_prev[state], value[state])
    return value


def Q_func_from_value(value):
    """TODO
    """
    q = defaultdict(dict)
    for state in possible_states:
        for action in actions:
            q[state][action] = sum(
                [transition_probability(state, action, next_state) *
                 (reward(state, action, next_state) + GAMMA * value[next_state])
                 for next_state in possible_states]
            )
    return q


def policy_iteration():
    """TODO

    """
    # turning lambda here into lookup
    policy = {state: np.array([0, 1]) for state in possible_states}
    for i in range(POLICY_ITERATIONS):
        value = value_from_policy(policy)
        q = Q_func_from_value(value)
        for state in possible_states:
            max_val = -1e13
            for action in actions:
                val = q[state][action]
                if val > max_val:
                    max_val = val
                    best_action = action
            policy[state] = best_action
    return policy


ACTION_TO_ACTION_CHAR = {
    (0, 1): ">",
    (0, -1): "<",
    (1, 0): "v",
    (-1, 0): "^",
}


def display_policy(policy):
    grid = np.array([
        ["", "", "", ""],
        ["", "", "", ""],
        ["", "", "", ""]
    ])
    for state, action in policy.items():
        grid[state[0], state[1]] = ACTION_TO_ACTION_CHAR[action]
    print(grid)


def main():
    policy = policy_iteration()
    display_policy(policy)


# = = = Tests = = =


def test_transition_probability():
    tests = [
        {
            "state": [2, 2],
            "action": [-1, 0],
            "next_states:truths": {
                (1, 2): .8,
                (2, 1): .1,
                (2, 3): .1,
                (1, 3):  0,
                (3, 3):  0
            }
        }, {
            "state": [0, 1],
            "action": [0, 1],
            "next_states:truths": {
                (0, 2): .8,
                (1, 1):  0,
                (-1, 1): 0
            }
        }, {
            "state": [0, 3],
            "action": [0, -1],
            "next_states:truths": {
                (0, 2): 0,
            }
        }
    ]
    for test in tests:
        state = np.array(test["state"])
        action = np.array(test["action"])
        for next_state, truth in test["next_states:truths"].items():
            next_state = np.array(next_state)
            # print("s, a, s'", state, action, next_state)
            result = transition_probability(state, action, next_state)
            nt.assert_equal(result, truth)


def test_compute_single_value():
    # previous_value = lambda s: -.1 if s == (1, 2) else 0
    previous_value = Counter({(1, 2): -1})
    # print("pv", previous_value)
    state = [0, 2]
    policy = lambda x: np.array([0, 1])
    result = compute_single_value(state, policy, previous_value)
    truth = 0.8 - GAMMA * .1
    nt.assert_almost_equal(result, truth)


def test_value_from_policy():
    global VALUE_ITERATIONS
    old_iterations = VALUE_ITERATIONS
    VALUE_ITERATIONS = 2
    policy = {state: np.array([0, 1]) for state in possible_states}
    result = value_from_policy(policy)
    truth = {(0, 0): 0, (0, 1): .64 * GAMMA}
    for state, truth_value in truth.items():
        nt.assert_almost_equal(result[state], truth_value)
    VALUE_ITERATIONS = old_iterations


def test_policy_iteration():
    global VALUE_ITERATIONS
    old_value_iterations = VALUE_ITERATIONS
    global POLICY_ITERATIONS
    old_policy_iterations = POLICY_ITERATIONS
    VALUE_ITERATIONS = 1
    POLICY_ITERATIONS = 1

    result = policy_iteration()
    truth = {(1, 2): (-1, 0)}
    for state, truth_action in truth.items():
        nt.assert_almost_equal(result[state], truth_action)

    POLICY_ITERATIONS = old_policy_iterations
    VALUE_ITERATIONS = old_value_iterations


if __name__ == "__main__":
    main()
