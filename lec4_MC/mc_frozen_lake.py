#!/usr/bin/python3
import numpy as np
import gym
import time
from mc_maze_agent import Agent

'''@brief transform the state value to a row and column in the environment
'''
def get_row_column(state):
    row = int(state/4)
    column = state - (row * 4)
    return row, column


'''@brief simulate the environment with the agents taught policy
'''
def simulate_episodes(agent, num_episodes=3):
    env2 = gym.make('FrozenLake-v1',  desc=None, render_mode="human", map_name="4x4", is_slippery=False)
    for _ in range(num_episodes):
        done = False
        state, _ = env2.reset()
        env2.render()
        while not done:
            # Random choice from behavior policy
            action = int(agent.pi[state])
            # take a step
            env2.render()
            time.sleep(0.1)
            next_state, _, done, _, _ = env2.step(action)
            state = next_state
        time.sleep(1.0)

def main():
    # MODE = ["LEARN"|"LOAD"]
    MODE = "LEARN"
    NUM_EPISODES = 50000

    # Define the Blackjack environment
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
    
    # Define the behavior policy (random policy for exploration)
    behavior_policy = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n

    # Define the target policy (random)
    target_policy = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n

    # Set hyperparameters
    gamma = 0.9
    action_desc = ["<", "|", ">", "^"]

    # Apply off-policy Monte Carlo algorithm
    agent = Agent(env, behavior_policy, target_policy, gamma)
    if MODE == "LEARN":
        Q, pi = agent.run_MC(NUM_EPISODES)
    elif MODE == "LOAD":
        # SAMPLE LEARNED POLICY -- Update after running for longer
        Q = np.array(
        [[0.531441,   0.59049   , 0.59049,    0.531441  ],
        [0.531441,   0.        , 0.6561 ,    0.59049   ],
        [0.59049 ,   0.72850953, 0.59049,    0.6561    ],
        [0.6561  ,   0.        , 0.59049,    0.59049   ],
        [0.59049 ,   0.6561    , 0.     ,    0.531441  ],
        [0.      ,   0.        , 0.     ,    0.        ],
        [0.      ,   0.81      , 0.     ,    0.6561    ],
        [0.      ,   0.        , 0.     ,    0.        ],
        [0.6561  ,   0.        , 0.729  ,    0.59049   ],
        [0.6561  ,   0.81      , 0.81   ,    0.        ],
        [0.729   ,   0.9       , 0.     ,    0.72855358],
        [0.      ,   0.        , 0.     ,    0.        ],
        [0.      ,   0.        , 0.     ,    0.        ],
        [0.      ,   0.80928255, 0.9    ,    0.729     ],
        [0.81    ,   0.9       , 1.     ,    0.81      ],
        [0.      ,   0.        , 0.     ,    0.        ]])
        agent.Q = Q
        agent.update_greedy_policy()
        pi = agent.pi
    # Print the estimated action-value function Q(s,a)
    print("Estimated Action-Value Function:")
    print(Q)

    # Print the target policy pi
    print("\nTarget Policy:")
    pi_mat = np.empty((env.action_space.n, env.action_space.n), dtype=str)
    for i, action in enumerate(pi):
        row, column = get_row_column(i)
        pi_mat[row][column] = action_desc[action]
    print(pi_mat)

    simulate_episodes(agent, num_episodes=3)

if __name__ == "__main__":
    main()