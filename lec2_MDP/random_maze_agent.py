#!/usr/bin/python3
import sys
import numpy as np
import random

import gym
import gym_maze

class Agent():
    def __init__(self, discount_factor=0.99, explore_rate=0.0, learning_rate=0.0):
        self.explore_rate = explore_rate
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
    
    def select_action(self, env, state):
        # Select a random action
        if random.random() < self.explore_rate:
            action = env.action_space.sample()
        else:
            action = self.policy(state, env)
        return action
    
    def policy(self, state, env):
        pass

    def update_policy(self, state_0, state, action, reward, learning_rate, discount_factor):
        pass

    def get_learning_rate(self, time):
        return self.learning_rate
    
    def get_explore_rate(self, time):
        return self.explore_rate
    
    def get_discount_factor(self, time):
        return self.discount_factor
        
        


def simulate():
    discount_factor = 0.99
    # Instantiate an agent
    agent = Agent(discount_factor, 1.0, 0.0)
    # Instantiating the learning related parameters
    learning_rate = agent.get_learning_rate(0)
    explore_rate = agent.get_explore_rate(0)

    num_streaks = 0

    # Render tha maze
    env.reset()
    env.render()

    for episode in range(NUM_EPISODES):

        # Reset the environment
        env.reset()
        solved = run_episode(agent, env, episode, explore_rate, learning_rate, discount_factor, num_streaks)
    
        if solved:
            num_streaks += 1
        else:
            num_streaks = 0

        # It's considered done when it's solved over 120 times consecutively
        if num_streaks > STREAK_TO_END:
            return 1

        # Update parameters
        explore_rate = agent.get_explore_rate(episode)
        learning_rate = agent.get_learning_rate(episode)

def run_episode(agent, env, episode, explore_rate, learning_rate, discount_factor=1.0, num_streaks=0):

        # Reset the environment
        obv = env.reset()

        # the initial state
        state_0 = state_to_bucket(obv)
        total_reward = 0

        for t in range(MAX_T):

            # Select an action
            # action = select_action(state_0, explore_rate)
            action = agent.select_action(env, state_0)

            # execute the action
            obv, reward, done, _, _ = env.step(action)

            # Observe the result
            state = state_to_bucket(obv)
            total_reward += reward

            # Update the current policy
            agent.update_policy(state_0, state, action, reward, learning_rate, discount_factor)

            # Updating the state for next iteration
            state_0 = state

            # Print debug data
            if DEBUG_MODE == 2:
                print("\nEpisode = %d" % episode)
                print("t = %d" % t)
                print("Action: %d" % action)
                print("State: %s" % str(state))
                print("Reward: %f" % reward)
                print("Best Q: %f" % best_q)
                print("Explore rate: %f" % explore_rate)
                print("Learning rate: %f" % learning_rate)
                print("Streaks: %d" % num_streaks)
                print("")

            elif DEBUG_MODE == 1:
                if done or t >= MAX_T - 1:
                    print("\nEpisode = %d" % episode)
                    print("t = %d" % t)
                    print("Explore rate: %f" % explore_rate)
                    print("Learning rate: %f" % learning_rate)
                    print("Streaks: %d" % num_streaks)
                    print("Total reward: %f" % total_reward)
                    print("")

            # Render tha maze
            if RENDER_MAZE:
                env.render()

            if env.is_game_over():
                sys.exit()

            if done:
                print("Episode %d finished after %f time steps with total reward = %f (streak %d)."
                      % (episode, t, total_reward, num_streaks))

                if t <= SOLVED_T:
                    return 1
                return 0

            elif t >= MAX_T - 1:
                print("Episode %d timed out at %d with total reward = %f."
                      % (episode, t, total_reward))

def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/bound_width
            scaling = (NUM_BUCKETS[i]-1)/bound_width
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)

if __name__ == "__main__":

    # Initialize the "maze" environment
    env = gym.make("maze-sample-5x5-v0")

    '''
    Defining the environment related constants
    '''
    # Number of discrete states (bucket) per state dimension
    MAZE_SIZE = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    NUM_BUCKETS = MAZE_SIZE  # one bucket per grid
    print("Number of buckets: ", NUM_BUCKETS)

    # Number of discrete actions
    NUM_ACTIONS = env.action_space.n  # ["N", "S", "E", "W"]
    print("Number of actions: ", NUM_ACTIONS)
    # Bounds for each discrete state
    STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))

    '''
    Defining the simulation related constants
    '''
    NUM_EPISODES = 100
    MAX_T = np.prod(MAZE_SIZE, dtype=int) * 100
    STREAK_TO_END = 10
    SOLVED_T = np.prod(MAZE_SIZE, dtype=int)
    DEBUG_MODE = 0
    RENDER_MAZE = True

    '''
    Begin simulation
    '''
    simulate()