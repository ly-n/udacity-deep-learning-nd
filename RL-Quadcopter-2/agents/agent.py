# TODO: your agent here!
import numpy as np
from task import Task

class PolicySearch_Agent():
    def __init__(self, task):
        # Task (environment) information
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low
        
        self.rotor_action_space = np.squeeze(self.create_uniform_grid([self.action_low], [self.action_high], \
                                                                      [self.action_range]))
        self.rotor_action_space_size = self.rotor_action_space.shape[0]
        #self.rotor_action_space = self.rotor_action_space.reshape(self.rotor_action_space_size,1)


        self.w = np.random.normal(
            size=(self.state_size, self.action_size),  # weights for simple linear policy: state_space x action_space
            scale=(self.action_range / (2 * self.state_size))) # start producing actions in a decent range

        # Score tracker and learning parameters
        self.best_w = None
        self.best_score = -np.inf
        self.noise_scale = 0.1

        # Episode variables
        self.reset_episode()
        
        ## Temporal Difference variables
        low = np.ones(self.state_size) * -30
        high = np.ones(self.state_size) * 30
        bins = np.ones(self.state_size) * 60
        self.grid = self.create_uniform_grid(low, high, bins)
        
        self.Q = np.zeros((self.grid.shape,self.action_size))
        self.episode = 0
        self.epsilon = 1

    def reset_episode(self):
        self.total_reward = 0.0
        self.count = 0
        state = self.task.reset()
        return state
    
    def epsilon_greedy_probs(self, state):
        discretized_state = self.discretize(state, self.grid)
        
        probs = np.ones((4,self.rotor_action_space_size)) * (self.epsilon / self.rotor_action_space_size)
        for i in range(self.action_size):
            best_a = np.argmax(self.Q[discretized_state][i]) # add noise later
            probs[i][best_a] = 1 - self.epsilon + (self.epsilon / (self.rotor_action_space_size))
        return probs

    def step(self, action, state, next_state, reward, done):
        # Save experience / reward
        self.total_reward += reward
        self.count += 1
        self.episode += 1
        self.epsilon = 1.0 / self.episode
        
        discretized_state = self.discretize(state, self.grid)
        discretized_next_state = self.discretize(next_state, self.grid)

        if not done:
            #self.learn()
            pass

        # Learn, if at end of episode
        if done:
            self.learn(action, state, next_state, reward)

    def act(self, state):
        #should return array of 4 rotor speeds
        # Choose action based on given state and policy
        discretized_state = self.discretize(state, self.grid)

        probs = self.epsilon_greedy_probs(discretized_state)
        #print('probs from agent.act')
        #print(probs.shape)
        #print(probs)
        #action = np.dot(state, self.w)  # simple linear policy

        action = []
        for i in range(self.action_size):
            action.append(np.random.choice(self.rotor_action_space, p=probs[i]))
 
        return action

    def learn(self, action, state, next_state, reward):
        #pick algo
        self.sarsa(action, state, next_state, reward)
        
        
        # Learn by random policy search, using a reward-based score
        self.score = self.total_reward / float(self.count) if self.count else 0.0
        if self.score > self.best_score:
            self.best_score = self.score
            self.best_w = self.w
            self.noise_scale = max(0.5 * self.noise_scale, 0.01)
        else:
            self.w = self.best_w
            self.noise_scale = min(2.0 * self.noise_scale, 3.2)
        self.w = self.w + self.noise_scale * np.random.normal(size=self.w.shape)  # equal noise in all directions
        
    def sarsa(self, action, state, next_state, reward):
        discretized_state = self.discretize(state, self.grid)
        discretized_next_state = self.discretize(next_state, self.grid)
        alpha = 1.0
        gamma = 0.99
        print(self.grid.shape)
        print(self.Q.shape)
        next_action = self.act(state)
        self.Q[discretized_state][action] = self.Q[discretized_state][action] + (alpha * (reward + (gamma*self.Q[discretized_next_state][next_action]) - self.Q[discretized_state][action]))
        
    def expected_sarsa(self, action, state, next_state, reward):
        discretized_state = self.discretize(state, self.grid)
        discretized_next_state = self.discretize(next_state, self.grid)
        alpha = 1.0
        gamma = 0.99
        
        next_action = self.select_action(state)
        
        self.Q[discretized_state][action] = self.Q[discretized_state][action] + (alpha * (reward + (gamma * np.dot(self.Q[discretized_next_state], next_probs)) - self.Q[discretized_state][action]))
        
     
    def create_uniform_grid(self, low, high, bins):
        spaces = []
        for i in range(len(low)):
            spaces.append(np.arange(float(low[i]), float(high[i]), abs(float(low[i])-float(high[i]))/float(bins[i])))
            spaces[i] = spaces[i][1:]

        return np.array(spaces)

    
    def discretize(self, sample, grid):
        return list(np.digitize(s, g) for s, g in zip(sample, grid))
        