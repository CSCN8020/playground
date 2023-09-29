import numpy as np
import gym
import time

# NOTE: This code is incomplete:
# TODO: Visualization
# TODO: Document methods

class Agent:
    def __init__(self, env, behavior_policy, target_policy, gamma ) -> None:
        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n
        self.Q =  np.zeros((self.num_states, self.num_actions))  # Initialize action-value function Q(s,a)
        self.C =  np.zeros((self.num_states, self.num_actions))  # Initialize cumulative sum of weights
        self.behavior_policy = behavior_policy
        self.target_policy = target_policy
        self.pi = np.ones(self.num_states, dtype=int)
        self.gamma = gamma
        self.env = env

    def run_MC(self, num_episodes):
        for episode in range(num_episodes):
            if episode % 10000 == 0:
                print("Episode: %d"%episode)

            # Generate an episode
            episode_states, episode_actions, episode_rewards = self.generate_episode()

            self.update_value_fn(episode_states, episode_actions, episode_rewards)
            
            self.update_greedy_policy()

        return self.Q, self.pi

    def generate_episode(self):
        episode_states = []
        episode_actions = []
        episode_rewards = []
        state, _ = self.env.reset()
        done = False
        while not done:
            # Random choice from behavior policy
            action = np.random.choice(range(self.num_actions), p=self.behavior_policy[state])

            # take a step
            next_state, reward, done, _, _ = self.env.step(action)
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            state = next_state

        return episode_states, episode_actions, episode_rewards
    
    def update_value_fn(self, episode_states, episode_actions, episode_rewards):
        G = 0  # Initialize the return
        W = 1  # Initialize the importance sampling ratio

        # Loop over the episode in reverse order
        for t in range(len(episode_states) - 1, -1, -1):
            state = episode_states[t]
            action = episode_actions[t]
            reward = episode_rewards[t]
            q = self.get_action_value(state, action)
            c = self.get_cum_sum_weights(state, action)
            G, q, c = self.policy_evaluation(G, q, c, W, reward)
            self.update_action_value(state, action, q)
            self.update_cum_sum_weights(state, action, c)
            self.update_greedy_policy()
            if action != self.pi[state]:  # If the action was not taken by the target policy
                break
            W /= self.query_behavior_policy_action(state, action)  # Update the importance sampling ratio

    def policy_evaluation(self, g, q, c, W, reward):
        g = self.gamma * g + reward  # Calculate the return
        c += W  # Update the cumulative sum of weights

        q += (W / c) * (g - q)  # Update action-value function
        return g, q, c
    
    def update_greedy_policy(self):
        # Calculate the target policy
        for s in range(self.num_states):
            best_action = np.argmax(self.Q[s])
            self.target_policy[s] = np.zeros((self.num_actions))
            self.target_policy[s, best_action] = 1
            self.pi[s] = best_action


    def get_action_value(self, state, action):
        return self.Q[state, action]
    
    def get_cum_sum_weights(self, state, action):
        return self.C[state, action]
    
    def update_action_value(self, state, action, q):
        self.Q[state, action] = q
    
    def update_cum_sum_weights(self, state, action, c):
        self.C[state, action] = c
                      
    def query_target_policy_action(self, state, action):
        return self.target_policy[state, action]
    
    def query_behavior_policy_action(self, state, action):
        return self.behavior_policy[state, action]

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