import numpy as np
import pandas as pd

GAMMA = 0.99
OBSTACLE = (2, 2)
PIT = (4, 2)
GOAL = (4, 3)

# Define the threshold for convergence
EPSILON= 0.025

# Transition probabilities for each action
TRANSITION_PROB = {
    'N': {'N': 0.8, 'E': 0.1, 'W': 0.1, 'S': 0},
    'S': {'N': 0, 'E': 0.1, 'W': 0.1, 'S': 0.8},
    'E': {'N': 0.1, 'E': 0.8, 'W': 0, 'S': 0.1},
    'W': {'N': 0.1, 'E': 0, 'W': 0.8, 'S': 0.1},
}

REWARDS = {
    (4, 3): 1,
    (4, 2): -1,
    "otherwise": -0.02,
}


def next_state(current_state, action):
    x, y = current_state

    # Define the possible moves and their effects on the state
    moves = {
        'N': (0, 1),
        'S': (0, -1),
        'E': (1, 0),
        'W': (-1, 0),
    }

    # Check if the current state is an absorbing state (goal or pit)
    if current_state == GOAL or current_state == PIT:
        return current_state

    # Sample the next move based on the transition probabilities
    next_move = np.random.choice(list(TRANSITION_PROB[action].keys()), p=list(TRANSITION_PROB[action].values()))

    # Calculate the next state based on the sampled move
    next_x, next_y = x + moves[next_move][0], y + moves[next_move][1]

    # If the next state is outside the map or in an obstacle, stay in the current state
    if next_x < 1 or next_x > 4 or next_y < 1 or next_y > 3 or (next_x, next_y) == OBSTACLE:
        return current_state

    return (next_x, next_y)

# Test the function with an example state and action 20 times
# for i in range(20):
#     print(next_state((3, 1), 'N'))

def value_iteration():
    V = np.zeros((4, 3))
    pi = np.empty((4, 3), dtype=str)

    while True:
        delta = 0
        for i in range(1, 5):
            for j in range(1, 4):
                s = (i, j)
                v = V[i - 1][j - 1]

                if s == GOAL or s == PIT or s == OBSTACLE:
                    continue
                action_values = []
                for a in TRANSITION_PROB.keys():
                    action_value = 0
                    for b in TRANSITION_PROB[a].keys():
                        next_s = next_state(s, b)
                        if next_s == GOAL:
                            r = REWARDS[GOAL]
                        elif next_s == PIT:
                            r = REWARDS[PIT]
                        else:
                            r = REWARDS["otherwise"]
                        action_value += TRANSITION_PROB[a][b] * (r + GAMMA * V[next_s[0] - 1][next_s[1] - 1])
                    action_values.append(action_value)

                max_action_value = max(action_values)
                V[i - 1][j - 1] = max_action_value
                
                # print(action_values.index(max_action_value))
                pi[i - 1][j - 1] = list(TRANSITION_PROB.keys())[action_values.index(max_action_value)]
                #print(v - V[i - 1][j - 1])
                delta = max(delta, abs(v - V[i - 1][j - 1]))

        print("Delta: ", delta)
        if delta < EPSILON:
            break

    return V, pi

V, pi = value_iteration()


# Print the optimal value function
print("Optimal Value Function:")
print(np.flipud(V.T))

# Print the optimal policy
print("Optimal Policy:")
print(np.flipud(pi.T))