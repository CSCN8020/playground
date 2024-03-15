import gym
import numpy as np
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

'''
    This code is modified from the Keras.io example found at: 
        https://keras.io/examples/rl/actor_critic_cartpole/
'''

class modelA2C:
    def __init__(self, num_inputs, num_hidden, num_actions):
        inputs = layers.Input(shape=(num_inputs,))
        common = layers.Dense(num_hidden, activation="relu")(inputs)
        # Actor: action
        action = layers.Dense(num_actions, activation="softmax")(common)
        # Outputs state value
        critic = layers.Dense(1)(common)

        self.model = keras.Model(inputs=inputs, outputs=[action, critic])

        self.optimizer = keras.optimizers.Adam(learning_rate=0.01)
        self.huber_loss = keras.losses.Huber()

    def act(self, state):
        # Predict action probabilities and estimated future rewards
        # from environment state
        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs, critic_value = self.model(state_tensor)

        # Sample action from action probability distribution
        action = np.random.choice(num_actions, p=np.squeeze(action_probs))
        return action_probs, critic_value, action
    
    '''@brief Return the value with the max probability
    '''
    def act_greedy(self, state):
        # Predict action probabilities and estimated future rewards
        # from environment state
        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor, 0)
        # Sample action from action probability distribution
        action_probs, _ = self.model(state_tensor)
        return np.argmax(action_probs)
    
    def update_model(self, history, tape):
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up recieving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret - value
            actor_losses.append(-log_prob * diff)  # actor loss

            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(
                self.huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )

        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename)


def test(model, ckpt, n_episodes=1, seed=42):
    env = gym.make("CartPole-v1", render_mode="human")  # Create the environment
    model.load_weights(ckpt)
    state, _ = env.reset(seed=seed)
    for episode in range(n_episodes):
        episode_reward = 0
        while True:  # Run with rendering
            env.render()
            time.sleep(0.05)
            action = model.act_greedy(state)
            state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            if done:
                print("Episode: {}, Reward: {}".format(episode, episode_reward))
                break
            

def train(model, ckpt = "", seed=42, max_steps_per_episode=10000, model_ckpt_name="", visualize_training=False):
    if visualize_training:
        env = gym.make("CartPole-v1", render_mode="human")  # Create the environment
    else:
        env = gym.make("CartPole-v1")
    
    if ckpt != "":
        model.load_weights(ckpt)
    
    action_probs_history = []
    critic_value_history = []
    rewards_history = []
    running_reward = 0
    episode_count = 0
    while True:  # Run until solved
        state, _ = env.reset(seed=seed)
        if visualize_training:
            env.render()
        episode_reward = 0
        with tf.GradientTape() as tape:
            for timestep in range(1, max_steps_per_episode):
                action_probs, critic_value, action = model.act(state)
                critic_value_history.append(critic_value[0, 0])
                action_probs_history.append(tf.math.log(action_probs[0, action]))

                # Apply the sampled action in our environment
                state, reward, done, _, _ = env.step(action)
                rewards_history.append(reward)
                episode_reward += reward

                if done:
                    break

            # Update running reward to check condition for solving
            running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

            # Calculate expected value from rewards
            # - At each timestep what was the total reward received after that timestep
            # - Rewards in the past are discounted by multiplying them with gamma
            # - These are the labels for our critic
            returns = []
            discounted_sum = 0
            for r in rewards_history[::-1]:
                discounted_sum = r + gamma * discounted_sum
                returns.insert(0, discounted_sum)

            # Normalize
            returns = np.array(returns)
            returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
            returns = returns.tolist()

            # Calculating loss values to update our network
            history = zip(action_probs_history, critic_value_history, returns)
            model.update_model(history, tape)

            # Clear the loss and reward history
            action_probs_history.clear()
            critic_value_history.clear()
            rewards_history.clear()

        # Log details
        episode_count += 1
        if episode_count % 10 == 0:
            template = "running reward: {:.2f} at episode {}"
            print(template.format(running_reward, episode_count))

        if running_reward > 195:  # Condition to consider the task solved
            print("Solved at episode {}!".format(episode_count))
            # if model_ckpt_name != "":
            #     model.save_weights(model_ckpt_name)
            break


if __name__=="__main__":
    # Configuration parameters for the whole setup
    MODE = "TRAIN"
    seed = 42
    gamma = 0.99  # Discount factor for past rewards
    max_steps_per_episode = 10000
    eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

    num_inputs = 4
    num_actions = 2
    num_hidden = 128
    model_name = "ckpt/model_cartpolev1.ckpt"
    visualize_training = True
    model = modelA2C(num_inputs, num_hidden, num_actions)
    if MODE == "TRAIN":
        train(model, seed=seed, max_steps_per_episode=max_steps_per_episode, 
              ckpt=model_name,
              visualize_training=visualize_training)
    elif MODE == "TEST":
        test(model, ckpt=model_name, n_episodes=3, seed=seed)