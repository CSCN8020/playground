import gym
from td_agents import SarsaAgent, QLAgent

def main(algorithm, num_episodes, gamma, alpha, epsilon):
    # Initialize the 'CliffWalking environment
    env = gym.make('CliffWalking-v0')
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
    env2 = gym.make('CliffWalking-v0', render_mode="human")
    agent.test(env2, 5, verbose=True)

if __name__ == "__main__":
    # Choose if you want to teach a QLearning/Sarsa agent
    algorithm = "QLearning"

    # Set hyperparameters
    gamma = 0.9     # discount factor
    alpha = 0.1     # learning rate
    epsilon = 0.3  # exploration factor

    # Set the number of episodes for learning
    num_episodes = 10000
    main(algorithm, num_episodes, gamma, alpha, epsilon)