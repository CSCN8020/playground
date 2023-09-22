import numpy as np

class GridWorld():
    def __init__(self, env_size, theta_threshold=0.01):
        self.env_size = env_size
        # Initialize the value function and set terminal state value to 0
        self.V = np.zeros((env_size, env_size))
        # TODO: Change the location of the terminal state and check how the optimal policy changes
        # TODO: Add more than one terminal state (requires more changes in the code)
        self.terminal_state = (4, 4)
        self.V[self.terminal_state] = 0

        self.theta_threshold = theta_threshold

        # Define the transition probabilities and rewards
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up
        self.action_description = ["Right", "Left", "Down", "Up"]
        self.gamma = 1.0  # Discount factor
        self.reward = -1  # Reward for non-terminal states
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
            next_i, next_j = self.step(action_index, i, j)
            if self.is_valid_state(next_i, next_j):
                value = self.get_value(next_i, next_j)
                if value >= max_value:
                    if value > max_value:
                        best_actions_str = self.action_description[action_index]
                    else:
                        best_actions_str += "|" + self.action_description[action_index]

                    best_action = action_index
                    max_value = value
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
    
    '''@brief use the Bellman equation to calculate the value of a single state
            Note that the equation is simplified due to the simple transition matrix
    '''
    def get_value(self, i, j):
        return self.reward + self.gamma * self.V[i, j]

    '''@brief Checks if there is the change in V is less than preset threshold
    '''
    def is_done(self, new_V):
        delta = abs(self.V - new_V)
        max_delta = delta.max()
        return max_delta <= self.theta_threshold
    
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

    '''@brief Finds the optimal action for every state and updates the policy
    '''
    def update_greedy_policy(self):
        # Note: We are assuming a greedy deterministic policy
        self.pi_str = []
        for i in range(self.env_size):
            pi_row = []
            for j in range(self.env_size):
                if self.is_terminal_state(i,j):
                    pi_row.append("NONE")
                    continue
                    
                _, self.pi_greedy[i,j], action_str = self.calculate_max_value(i, j)
                pi_row.append(action_str)
            self.pi_str.append(pi_row)
        
    '''@brief Returns the stored greedy policy
    '''
    def get_policy(self):
        return self.pi_greedy
    
    '''@brief Prints the policy using the action descriptions
    '''
    def print_policy(self):
        for row in self.pi_str:
            print(row)

def main():
    ENV_SIZE = 5
    THETA_THRESHOLD = 0.05
    MAX_ITERATIONS = 1000
    gridworld = GridWorld(ENV_SIZE, THETA_THRESHOLD)
    
    # Perform value iteration
    done = False
    for iter in range(MAX_ITERATIONS):
        # Add stopping criteria if change in value function is small
        if done: break
        # Make a copy of the value function
        # TODO: Try in-place state-value function update where Vpi is updated with every state
        new_V = np.copy(gridworld.get_value_function())
        # Loop over all states
        for i in range(ENV_SIZE):
            for j in range(ENV_SIZE):
                if not gridworld.is_terminal_state(i, j):
                    new_V[i, j], _, _= gridworld.calculate_max_value(i,j)
        # TODO: Uncomment the next line and compare how many iterations it takes
        # TODO: Change the theta_threshold value to a large value (1.0) and explore what happens to the optimal state-value function and policy
        # done = gridworld.is_done(new_V)
        gridworld.update_value_function(new_V)

    # Print the optimal value function
    print("Optimal Value Function Found in %d iterations:"%(iter+1))
    print(gridworld.get_value_function())

    gridworld.update_greedy_policy()
    gridworld.print_policy()

if __name__=="__main__":
    main()
