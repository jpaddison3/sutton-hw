"""Sutton and Barto, Exercise 4.4

Goal: Create and MDP from the problem statement, with interpretation, then solve.
"""
import itertools
from collections import Counter, defaultdict
import math
import numpy as np
import nose.tools as nt

# = = = MDP Definition = = =
max_cars = 20
possible_cars_A = list(range(0, max_cars + 1))
possible_cars_B = list(range(0, max_cars + 1))
# TODO
possible_states = list(itertools.product(possible_cars_A, possible_cars_B))
# TODO
actions = list(range(-5, 6))
transitions_A = {}
transitions_B = {}


def poisson(expected_number, n, cutoff):
    if n == cutoff:
        return 1 - sum([poisson(expected_number, i, cutoff) for i in range(0, cutoff)])
    return expected_number ** n / math.factorial(n) * np.e**(-expected_number)


def construct_transitions(expected_rentals, expected_returns):
    """TODO

    returns {(cars, next_cars): (weighted_reward, probability)}
    """
    transitions = defaultdict(lambda: [0, 0])
    for cars in range(0, max_cars + 1):
        # print("~ ~ ~ cars", cars)
        for rentals in range(0, cars + 1):
            # print("~ ~ rentals", rentals)
            prob_rentals = poisson(expected_rentals, rentals, cars)
            max_returns = max_cars - (cars - rentals)
            for returns in range(0, max_returns + 1):
                # print("~ returns", returns)
                next_cars = cars - rentals + returns
                reward = 10 * rentals
                prob_returns = poisson(expected_returns, returns, max_returns)
                # print("next_cars reward, prob_rentals, prob_returns",
                #       next_cars, reward, prob_rentals, prob_returns)
                transitions[(cars, next_cars)][0] += reward * prob_rentals * prob_returns
                transitions[(cars, next_cars)][1] += prob_rentals * prob_returns
        # The reason for this .. is complicated
        for next_cars in range(0, max_cars + 1):
            transitions[(cars, next_cars)][0] /= transitions[(cars, next_cars)][1]
    return transitions


def construct_mdp_transitions():
    global transitions_A, transitions_B
    transitions_A = construct_transitions(3, 3)
    transitions_B = construct_transitions(4, 2)


def reward(state, action, next_state):
    """TODO

    """
    reward = -2 * abs(action)
    if action > 0:
        reward += 2
    if state[0] - action > 10:
        reward -= 4 * (state[0] - action - 10)
    if state[1] + action > 10:
        reward -= 4 * (state[1] + action - 10)
    reward += transitions_A[state[0] - action, next_state[0]][0] + \
        transitions_B[state[1] + action, next_state[1]][0]
    return reward


# TODO Maybe not this either
def possible_next_states(state):
    """TODO

    """
    pass


def transition_probability(state, action, next_state):
    """TODO

    """
    return transitions_A[state[0] - action, next_state[0]][1] * \
        transitions_B[state[1] + action, next_state[1]][1]


# = = = Solver = = =
GAMMA = .9
VALUE_ITERATIONS = 12
POLICY_ITERATIONS = 12


def compute_single_value(state, policy, previous_value):
    """TODO

    """
    action = policy[state]
    # print("a", action)
    value = 0
    for next_state in possible_states:
        prob = transition_probability(state, action, next_state)
        got_reward = reward(state, action, next_state)
        # if prob != 0:
        #     print("s', p, r, V(s'):", next_state, prob, got_reward, previous_value[next_state])
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


def is_in_bounds(state, action):
    return state[0] - action >= 0 and state[0] - action <= max_cars and state[1] + \
        action >= 0 and state[1] + action <= max_cars


def Q_func_from_value(value):
    """TODO
    """
    q = defaultdict(dict)
    for state in possible_states:
        for action in actions:
            if is_in_bounds(state, action):
                q[state][action] = sum(
                    [transition_probability(state, action, next_state) *
                     (reward(state, action, next_state) + GAMMA * value[next_state])
                     for next_state in possible_states]
                )
            else:
                # No outside bounds allowed
                q[state][action] = -1e14
    return q


def policy_iteration():
    """TODO

    """
    # turning lambda here into lookup
    policy = {state: 0 for state in possible_states}
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


def display_policy(policy):
    grid = np.zeros((20, 20)).astype("int")
    for state, action in policy.items():
        grid[state[0] - 1, state[1] - 1] = action
    print(grid)


def main():
    construct_mdp_transitions()
    policy = policy_iteration()
    display_policy(policy)


# = = = Tests = = =


def test_poisson():
    tests = [
        {
            "expected_number": 1,
            "n": 1,
            "cutoff": 5,
            # These truth values are ascertained to be sane, but not checked exactly
            "truth": 0.36787944117144233,
        }, {
            "expected_number": 10,
            "n": 10,
            "cutoff": 10,
            "truth": 0.5420702855281476,
        }, {
            "expected_number": 2,
            "n": 0,
            "cutoff": 0,
            "truth": 1,
        }
    ]
    for test in tests:
        result = poisson(test["expected_number"], test["n"], test["cutoff"])
        nt.assert_almost_equal(result, test["truth"])


def test_construct_transitions():
    global max_cars
    old_max_cars = max_cars
    max_cars = 5

    transitions = construct_transitions(4, 2)
    truth = {
        (5, 0): [50, poisson(4, 5, 5) * poisson(2, 0, 5)]
    }
    for state_change, transition_truth in truth.items():
        np.testing.assert_array_almost_equal(transitions[state_change], transition_truth)

    max_cars = old_max_cars


# TODO Remove?
def test_reward():
    global transitions_A, transitions_B
    transitions_A = defaultdict(lambda: [0, 0])
    transitions_A[(0, 0)] = [0, 0]
    transitions_A[(2, 0)] = [30, .3]
    transitions_A[(11, 0)] = [110, .1]
    transitions_B = defaultdict(lambda: [0, 0])
    transitions_B[(0, 0)] = [0, 0]
    transitions_B[(2, 0)] = [30, .2]
    tests = [
        {
            "state": (2, 0),
            "action": 2,
            "next_states:truths": {
                (0, 0): 28,
            }
        }, {
            "state": (0, 2),
            "action": -2,
            "next_states:truths": {
                (0, 0): 26,
            }
        }, {
            "state": (11, 0),
            "action": 0,
            "next_states:truths": {
                (0, 0): 106,
            }
        }
    ]
    for test in tests:
        state = np.array(test["state"])
        action = np.array(test["action"])
        for next_state, truth in test["next_states:truths"].items():
            next_state = np.array(next_state)
            # print("s, a, s'", state, action, next_state)
            result = reward(state, action, next_state)
            # print("result , truth", result, truth)
            nt.assert_equal(result, truth)


# TODO Remove?
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
    global transitions_A, transitions_B
    transitions_A = defaultdict(lambda: [0, 0])
    transitions_A[(0, 0)] = [0, 0]
    transitions_A[(1, 0)] = [10, .3]
    transitions_B = defaultdict(lambda: [0, 0])
    transitions_B[(0, 0)] = [0, 0]
    transitions_B[(1, 0)] = [10, .2]

    previous_value = Counter({(0, 0): -1})
    state = (1, 1)
    policy = defaultdict(lambda: 0)
    result = compute_single_value(state, policy, previous_value)
    truth = 0.06 * (20 - GAMMA * 1)
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


def test_is_in_bounds():
    tests = [
        {
            "state": (0, 0),
            "action": 1,
            "truth": False
        }, {
            "state": (0, 20),
            "action": 1,
            "truth": False
        }, {
            "state": (0, 19),
            "action": 1,
            "truth": True,
        }, {
            "state": (0, 0),
            "action": 0,
            "truth": True,
        }
    ]


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
