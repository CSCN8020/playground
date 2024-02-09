import numpy as np
import gym
import time

class TDAgent:
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration factor
        self.action_desc = ["<", "|", ">", "^"]

        # Initialize Q-values for all state-action pairs
        self.q_values = np.zeros((num_states, num_actions))

    '''@brief Chooses an action using the current policy
    '''
    def select_action(self, state):
        # Epsilon-greedy policy
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)  # Random action with probability epsilon
        else:
            return self.select_greedy_action(state) # Greedy action with probability (1 - epsilon)

    '''@brief Returns the action that corresponds to the greedy policy
    '''
    def select_greedy_action(self, state):
        q_values_state = self.q_values[state]
        return np.argmax(q_values_state) 

    '''@brief Updates the action-value function using 
        inputs: current state, action, reward, and next state and action
    '''
    def update_q_values(self, state, action, reward, next_state, next_action):
        pass

    '''@brief Returns the q_values
    '''
    def get_q_values(self):
        return self.q_values

    '''@brief training loop for Sarsa algorithm
    '''
    def train(self, env, num_episodes):
        for episode in range(num_episodes):
            if episode % 1000 == 0:
                print("Episode: %d"%episode)

            state, _ = env.reset()
            action = self.select_action(state)
            while True:
                next_state, reward, done, _, _ = env.step(action)
                next_action = self.select_action(next_state)

                self.update_q_values(state, action, reward, next_state, next_action)

                if done:
                    break

                state = next_state
                action = next_action

    '''@brief Run the simulation with the taught policy and rendering (if activated in env)
    '''
    def test(self, env, num_episodes=1, verbose=False):
        for _ in range(num_episodes):
            done = False
            state, _ = env.reset()
            env.render()
            while not done:
                # Random choice from behavior policy
                action = self.select_greedy_action(state)
                # Render environment and pause
                env.render()
                time.sleep(0.1)
                # take a step 
                if verbose:
                    print("Moving: %s"%self.action_desc[action])
                next_state, _, done, _, _ = env.step(action)
                state = next_state
            time.sleep(1.0)

class SarsaAgent(TDAgent):
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        super().__init__(num_states, num_actions, alpha, gamma, epsilon)

    '''@brief Updates the action-value function using 
        inputs: current state, action, reward, and next state and action
    '''
    def update_q_values(self, state, action, reward, next_state, next_action):
        # TODO: Apply SARSA update rule to update Q(s,a)


class QLAgent(TDAgent):
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        super().__init__(num_states, num_actions, alpha, gamma, epsilon)
    
    '''@brief Updates the action-value function using 
        inputs: current state, action, reward, and next state and action
    '''
    def update_q_values(self, state, action, reward, next_state, next_action):
        # TODO: Apply QLearning update rule to update Q(s,a)