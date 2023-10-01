#!/usr/bin/python3

import numpy as np

ENV_SIZE = 5

class GridWorld():

    def __init__(self, env_size):
        self.env_size = env_size
        # Initialize the value function and set terminal state value to 0
        self.V = np.zeros((env_size, env_size))
        self.terminal_state = (4, 4)
        self.V[self.terminal_state] = 0

        # Define the transition probabilities and rewards
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up
        self.action_description = ["Right", "Left", "Down", "Up"]
        self.gamma = 1.0  # Discount factor
        self.reward = -1  # Reward for non-terminal states
        self.pi_greedy = np.zeros((self.env_size, self.env_size), dtype=int)
    
    '''@brief Checks if there is the change in V is less than preset threshold
    '''
    def is_done(self, i, j):
        pass
    
    '''@brief Returns True if the state is a terminal state
    '''
    def is_terminal_state(self, i, j):
        return (i, j) == self. terminal_state
    
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

    '''@brief Returns the stored greedy policy
    '''
    def get_policy(self):
        return self.pi_greedy
    
    '''@brief Prints the policy using the action descriptions
    '''
    def print_policy(self):
        for row in self.pi_str:
            print(row)

    '''@brief Calculate the maximim value by following a greedy policy
    '''
    def calculate_max_value(self, i, j):
        # TODO: Find the maximum value for the current state using Bellman's equation
        # HINT #1 start with a - infinite value as the max
        max_value = float('-inf')
        best_action = None
        best_actions_str = ""
        # HINT #2: Loop over all actions
        for action_index in range(len(self.actions)):
          # TODO: Find Next state
          next_i, next_j = self.step(action_index, i, j)
          if self.is_valid_state(next_i, next_j):
              pass
            # Calculate value function if state is valid
                
            # TODO: Update the max_value as required
            
          '''
            TODO: Optional - You can also update the best action and best_actions_str to update the policy
            Otherwise, feel free to change the return values and add any extra methods to calculate the greedy policy
          '''
          pass


        return max_value, best_action, best_actions_str
    
    '''@brief Returns the next state given the chosen action and current state
    '''
    def step(self, action_index, i, j):
        # We are assuming a Transition Probability Matrix where
        # P(s'|s) = 1.0 for a single state and 0 otherwise
        action = self.actions[action_index]
        return i + action[0], j + action[1]
    
    '''@brief Checks if a state is within the acceptable bounds of the environment
    '''
    def is_valid_state(self, i, j):
        valid = 0 <= i < self.env_size and 0 <= j < self.env_size
        return valid
    
    def update_greedy_policy(self):
        for i in range(ENV_SIZE):
            for j in range(ENV_SIZE):
                # TODO: calculate the greedy policy and populate self.pi_greedy

                # TODO: Optional - Add the optimal action description to self.pi_str to be able to print it
                pass
        


gridworld = GridWorld(ENV_SIZE)
# Perform value iteration
num_iterations = 1000

for _ in range(num_iterations):
    pass
    # TODO: Add another stopping criteria (implement in GridWorld's is_done)
    # TODO: Make a copy of the value function
    # TODO: For all states, update the *copied* value function using GridWorld's calculate_max_value
    # LOOP GOES HERE

    # TODO: After updating all states, update the value function using GridlWorld's update_value_function

# Print the optimal value function
print("Optimal Value Function:")
print(gridworld.get_value_function())

gridworld.update_greedy_policy()
gridworld.print_policy()

