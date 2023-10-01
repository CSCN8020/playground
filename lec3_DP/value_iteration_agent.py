#!/usr/bin/python3

import numpy as np
from gridworld import GridWorld

class Agent():
    def __init__(self, env, theta_threshold=0.01):
        self.env_size = env.get_size()
        self.env = env
        # Initialize the value function and set terminal state value to 0
        self.V = np.zeros((self.env_size, self.env_size))
        # TODO: Change the location of the terminal state and check how the optimal policy changes
        # TODO: Add more than one terminal state (requires more changes in the code)
        self.terminal_state = (4, 4)
        self.V[self.terminal_state] = 0

        self.theta_threshold = theta_threshold

        # Define the transition probabilities and rewards
        self.actions = env.get_actions()  # Right, Left, Down, Up
        self.gamma = 1.0  # Discount factor
        self.pi_greedy = np.zeros((self.env_size, self.env_size), dtype=int)

    '''@brief Calculate the maximim value by following a greedy policy
    '''
    def calculate_max_value(self, i, j):
        # Find the maximum value for the current state using Bellman's equation
        # Start with a - infinite value as the max
        max_value = float('-inf')
        best_action = None
        best_actions_str = ""
        # Loop over all actions
        for action_index in range(len(self.actions)):
          # Find Next state
            next_i, next_j, reward, _ = self.env.step(action_index, i, j)
            if self.env.is_valid_state(next_i, next_j):
                value = self.get_value(next_i, next_j, reward)
                if value >= max_value:
                    # Populating the best_actions description string
                    if value > max_value:
                        best_actions_str = self.env.action_description[action_index]
                    else:
                        best_actions_str += "|" + self.env.action_description[action_index]

                    best_action = action_index
                    max_value = value
        return max_value, best_action, best_actions_str
    
    '''@brief use the Bellman equation to calculate the value of a single state
            Note that the equation is simplified due to the simple transition matrix
    '''
    def get_value(self, i, j, reward):
        return reward + self.gamma * self.V[i, j]
    
    '''
    @brief Overwrites the current state-value function with a new one
    '''
    def update_value_function(self, V):
        self.V = np.copy(V)

    '''
    @brief Returns the full state-value function V_pi
    '''
    def get_value_function(self):
        return self.V

    '''@brief Finds the optimal action for every state and updates the policy
    '''
    def update_greedy_policy(self):
        # Note: We are assuming a greedy deterministic policy
        self.pi_str = []
        for i in range(self.env_size):
            pi_row = []
            for j in range(self.env_size):
                if self.env.is_terminal_state(i,j):
                    pi_row.append("X")
                    continue
                    
                _, self.pi_greedy[i,j], action_str = self.calculate_max_value(i, j)
                pi_row.append(action_str)
            self.pi_str.append(pi_row)
        
    '''@brief Checks if there is the change in V is less than preset threshold
    '''
    def is_done(self, new_V):
        delta = abs(self.V - new_V)
        max_delta = delta.max()
        return max_delta <= self.theta_threshold
    
    '''@brief Returns the stored greedy policy
    '''
    def get_policy(self):
        return self.pi_greedy
    
    '''@brief Prints the policy using the action descriptions
    '''
    def print_policy(self):
        for row in self.pi_str:
            print(row)