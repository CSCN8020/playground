import gym
from td_agents import SarsaAgent, QLAgent

def main(algorithm, num_episodes, gamma, alpha, epsilon):
    # Initialize the FrozenLake environment
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
    # Define the number of states and actions
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # Initialize the Sarsa agent
    if algorithm == "Sarsa":
        agent = SarsaAgent(num_states, num_actions, alpha, gamma, epsilon)
    elif algorithm == "QLearning":
        agent = QLAgent(num_states, num_actions, alpha, gamma, epsilon)
    else:
        print("ERROR: Incorrect algorithm name: %s. Please specify one of: ['QLearning', 'Sarsa']"%algorithm)
    
    # Start the training
    print("Training %s agent for %d episodes."%(algorithm, num_episodes))
    agent.train(env, num_episodes)

    # Test (and visualize the learnt policys)
    env2 = gym.make('FrozenLake-v1', desc=None, render_mode="human", map_name="4x4", is_slippery=False)
    agent.test(env2, 2, verbatim=True)

if __name__ == "__main__":
    # Choose if you want to teach a QLearning/Sarsa agent
    algorithm = "Sarsa"

    # Set hyperparameters
    gamma = 0.9     # discount factor
    alpha = 0.1     # learning rate
    epsilon = 0.3   # exploration factor

    # Set the number of episodes for learning
    num_episodes = 5000
    main(algorithm, num_episodes, gamma, alpha, epsilon)