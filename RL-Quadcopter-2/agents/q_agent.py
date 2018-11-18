# TODO: your agent here!
import numpy as np
from task import Task
from collections import deque
import tensorflow as tf

class q_agent():
    def __init__(self, task, learning_rate=0.01, hidden_size=10, name='QNetwork'):
        # Task (environment) information
        with tf.Graph().as_default() as self.graph:
            self.task = task
            self.state_size = task.state_size
            self.action_size = task.action_size
            self.action_low = task.action_low
            self.action_high = task.action_high
            self.action_range = self.action_high - self.action_low

            self.rotor_action_space = np.squeeze(self.create_uniform_grid([self.action_low], [self.action_high], \
                                                                          [self.action_range]))
            self.rotor_action_space_size = self.rotor_action_space.shape[0]

            self.episode = 0
            self.epsilon = 1
            self.total_reward = 0.0

            self.learning_rate=learning_rate
            self.hidden_size=hidden_size
            self.inputs_ = tf.placeholder(tf.float32, [None, self.state_size], name='inputs')

            # One hot encode the actions to later choose the Q-value for the action
            self.actions_ = tf.placeholder(tf.int32, [None, None], name='actions')
            one_hot_actions = tf.one_hot(self.actions_, (self.rotor_action_space_size * self.action_size))

            # Target Q values for training
            self.targetQs_ = tf.placeholder(tf.float32, [None], name='target')

            # ReLU hidden layers
            self.fc1 = tf.contrib.layers.fully_connected(self.inputs_, self.hidden_size)
            self.fc2 = tf.contrib.layers.fully_connected(self.fc1, self.hidden_size)

            # Linear output layer
            self.output = tf.contrib.layers.fully_connected(self.fc2, (self.rotor_action_space_size * self.action_size), 
                                                            activation_fn=None)

            ### Train with loss (targetQ - Q)^2
            # output has length 2, for two actions. This next line chooses
            # one value from output (per row) according to the one-hot encoded actions.
            self.Q = tf.reduce_sum(tf.multiply(self.output, one_hot_actions), axis=1)

            self.loss = tf.reduce_mean(tf.square(self.targetQs_ - self.Q))
            self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        

    def reset_episode(self):
        self.total_reward = 0.0
        self.count = 0
        state = self.task.reset()
        return state
    
    
    def sample(self):
        results = []
        for i in range(4):
            results.append(np.random.choice(self.rotor_action_space))
        return results
    
    def create_uniform_grid(self, low, high, bins):
        spaces = []
        for i in range(len(low)):
            spaces.append(np.arange(float(low[i]), float(high[i]), abs(float(low[i])-float(high[i]))/float(bins[i])))
            spaces[i] = spaces[i][1:]

        return np.array(spaces)

    
    def discretize(self, sample, grid):
        return list(np.digitize(s, g) for s, g in zip(sample, grid))
    

    

class Memory():
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), 
                               size=batch_size, 
                               replace=False)
        return [self.buffer[ii] for ii in idx]

        
    

