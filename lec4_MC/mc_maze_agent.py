#!/usr/bin/python3
import numpy as np

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

    '''@brief Runs the Incremental Off-policy Monte Carlo Algorithm with Importance Sampling
    '''
    def run_MC(self, num_episodes):
        for episode in range(num_episodes):
            if episode % 10000 == 0:
                print("Episode: %d"%episode)

            # Generate an episode
            episode_states, episode_actions, episode_rewards = self.generate_episode()

            self.update_value_fn(episode_states, episode_actions, episode_rewards)
            
            self.update_greedy_policy()

        return self.Q, self.pi

    '''@brief Generates a full episode following the behavior policy and returns the sequence of states, actions, and rewards
    '''
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
    
    '''@brief Update the value function using incremental MC with importance sampling
    '''
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
            # Update stage
            self.update_action_value(state, action, q)
            self.update_cum_sum_weights(state, action, c)
            self.update_greedy_policy()
            if action != self.pi[state]:  # If the action was not taken by the target policy
                break
            W /= self.query_behavior_policy_action(state, action)  # Update the importance sampling ratio

    '''@brief Evaluate the return and action-value
    '''
    def policy_evaluation(self, g, q, c, W, reward):
        g = self.gamma * g + reward  # Calculate the return
        c += W  # Update the cumulative sum of weights

        q += (W / c) * (g - q)  # Update action-value function
        return g, q, c
    
    '''@brief Update the target policy
    '''
    def update_greedy_policy(self):
        # Calculate the target policy
        for s in range(self.num_states):
            best_action = np.argmax(self.Q[s])
            self.target_policy[s] = np.zeros((self.num_actions))
            self.target_policy[s, best_action] = 1
            self.pi[s] = best_action

    '''@brief Retrieve the action value of the current (state, action) pair
        @note These methods can be changed for different matrix shapes without changing the core code
    '''
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
