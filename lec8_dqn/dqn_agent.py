import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam

class DQN_Agent:
    #
    # Initializes attributes and constructs CNN model and target_model
    #
    def __init__(self, state_size, action_size, gamma:float = 0.985, memory_size:int = 10000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        
        # Hyperparameters
        self.gamma = gamma            # Discount rate
        self.epsilon = 1.0          # Exploration rate
        self.epsilon_min = 0.05      # Minimal exploration rate (epsilon-greedy)
        self.epsilon_decay = 0.985  # Decay rate for epsilon
        self.update_rate = 10000     # Number of steps until updating the target network
        
        # Construct DQN models
        # gpus = tf.config.list_physical_devices('GPU')
        # if gpus:
        #     # Restrict TensorFlow to only use the first GPU
        #     try:
        #         tf.config.set_visible_devices(gpus[0], 'GPU')
        #         logical_gpus = tf.config.list_logical_devices('GPU')
        #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        #     except RuntimeError as e:
        #         # Visible devices must be set before GPUs have been initialized
        #         print(e)
        # else:
        #     print("No GPUs")
        self.model = self._build_model() # Q Network
        self.target_model = self._build_model() # Target Network
        self.target_model.set_weights(self.model.get_weights())
        self.model.summary()

    #
    # Constructs CNN
    #
    def _build_model(self):
        model = Sequential()
        
        # Conv Layers
        model.add(Conv2D(32, (8, 8), strides=4, padding='same', input_shape=self.state_size))
        model.add(Activation('relu'))
        
        model.add(Conv2D(64, (4, 4), strides=2, padding='same'))
        model.add(Activation('relu'))
        
        model.add(Conv2D(64, (3, 3), strides=1, padding='same'))
        model.add(Activation('relu'))
        model.add(Flatten())

        # FC Layers
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        
        model.compile(loss='huber_loss', optimizer=Adam())
        return model

    #
    # Stores experience in replay memory
    #
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    #
    # Chooses action based on epsilon-greedy policy
    #
    def act(self, state):
        # Random exploration
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(state, verbose=0)
        
        return np.argmax(act_values[0])  # Returns action using policy
    
    def greedy_act(self, state):
        act_values = self.model.predict(state, verbose=0)
        
        return np.argmax(act_values[0])  # Returns action using policy
    
    

    #
    # Trains the model using randomly selected experiences in the replay memory
    #
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            
            if not done:
                target = (reward + self.gamma * np.max(self.target_model.predict(next_state, verbose=0)))
            else:
                target = reward
                
            # Construct the target vector as follows:
            # 1. Use the current model to output the Q-value predictions
            target_f = self.model.predict(state, verbose=0)
            
            # 2. Rewrite the chosen action value with the computed target
            target_f[0][action] = target
            
            # 3. Use vectors in the objective computation
            self.model.fit(state, target_f, epochs=1, verbose=0)
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    #
    # Sets the target model parameters to the current model parameters
    #
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
            
    #
    # Loads a saved model
    #
    def load(self, name):
        self.model.load_weights(name)
        self.target_model.load_weights(name)

    #
    # Saves parameters of a trained model
    #
    def save(self, name):
        self.model.save_weights(name)