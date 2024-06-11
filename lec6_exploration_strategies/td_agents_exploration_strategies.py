import numpy as np
import time
import random

class TDAgent:
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.9, epsilon=0.1, exploration_approach="UCB"):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration factor
        # self.action_desc = ["<", "|", ">", "^"]
        self.exploration_approach = exploration_approach # UCB/epsilon-greedy
        self.c = 0.5
        self.action_counts = np.zeros((num_states, num_actions))
        self.num_steps = 0

        # Initialize Q-values for all state-action pairs
        self.q_values = np.zeros((num_states, num_actions))

    '''@brief Chooses an action (index) using the current policy
    '''
    def select_action(self, state) -> int:
        if self.exploration_approach == "UCB":
            return self.upper_confidence_bound(state)
        elif self.exploration_approach == "epsilon-greedy":
            return self.epsilon_greedy(state)
        else:
            raise("Error: Unknown exploration approach")

    def upper_confidence_bound(self, state) -> int:
        qmax = float("-inf")
        best_action = None
        for action in range(0, self.num_actions):
            action_repetitions = self.action_counts[state, action]
            if action_repetitions == 0:
                # print(f'Never tried action : {action}')
                return action
            q_ucb = self.q_values[state, action] + self.c * np.sqrt( np.log(self.num_steps) / action_repetitions )
            if q_ucb > qmax:
                qmax = q_ucb
                best_action = action
        return best_action

    def epsilon_greedy(self, state) -> int:
        # Generate random value between 0 and 1
        p = np.random.rand()

        # Compare random value to self.epsilon and choose greedy vs random action accordingly 
        if p <= self.epsilon:
            return random.randint(0,self.num_actions-1)
            # return np.random.choice(self.num_actions, p=[1/self.num_actions]*self.num_actions)
        return self.select_greedy_action(state)


    '''@brief Returns the action that corresponds to the greedy policy
    '''
    def select_greedy_action(self, state) -> int:
        # Return the action that maximizes the q-value at the current state
        # Loop through all actions and compare the Q value and find the max
        return np.argmax(self.q_values[state,:])

    '''@brief Updates the action-value function using 
        inputs: current state, action, reward, and next state and action
        Implementation will vary between SARSA and QLearning
    '''
    def update_q_values(self, state, action, reward, next_state, next_action) -> None:
        pass

    '''@brief Returns the q_values
    '''
    def get_q_values(self):
        return self.q_values

    '''@brief training loop for Sarsa algorithm
    '''
    def train(self, env, num_episodes):
        self.num_steps = 0
        for episode in range(num_episodes):
            # print(self.action_counts)
            if episode % 1000 == 0:
                print("Episode: %d"%episode)
                # print(self.get_q_values())

            state, _ = env.reset()
            action = self.select_action(state)
            while True:
                next_state, reward, done, _, _ = env.step(action)
                self.num_steps += 1
                self.action_counts[state, action] += 1 
                next_action = self.select_action(next_state)

                self.update_q_values(state, action, reward, next_state, next_action)

                if done:
                    break
                state = next_state
                action = next_action  
        print(self.action_counts)

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
                # if verbose:
                #     print("Moving: %s"%self.action_desc[action])
                next_state, _, done, _, _ = env.step(action)
                state = next_state
            time.sleep(1.0)
            
class SarsaAgent(TDAgent):
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.9, epsilon=0.1, exploration_approach="UCB"):
        super().__init__(num_states, num_actions, alpha, gamma, epsilon, exploration_approach)

    '''@brief Updates the action-value function using 
        inputs: current state, action, reward, and next state and action
    '''
    def update_q_values(self, state, action, reward, next_state, next_action) -> None:
        # Apply SARSA update rule to update Q(s,a)
        self.q_values[state, action] += self.alpha*(reward + self.gamma*self.q_values[next_state, next_action] - self.q_values[state, action])

class QLAgent(TDAgent):
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.9, epsilon=0.1, exploration_approach="UCB"):
        super().__init__(num_states, num_actions, alpha, gamma, epsilon, exploration_approach)
    
    '''@brief Updates the action-value function using 
        inputs: current state, action, reward, and next state and action
    '''
    def update_q_values(self, state, action, reward, next_state, next_action=None) -> None:
        next_q = np.max(self.q_values[next_state,:])
        self.q_values[state, action] += self.alpha*(reward + self.gamma*next_q - self.q_values[state, action])