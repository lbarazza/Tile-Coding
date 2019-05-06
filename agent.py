from tiles import *

class Agent:
    def __init__(self, env_info, tiling_specs, alpha=0.1, gamma=0.99):

        self.nA = env_info[0]
        self.low_observation_space = env_info[1]
        self.high_observation_space = env_info[2]
        self.Q = TiledQTable(self.low_observation_space, self.high_observation_space, tiling_specs, self.nA)

        self.alpha = alpha
        self.gamma = gamma
        self.nEpisode = 1
        self.epsilon = 1.0/self.nEpisode


    def choose_action(self, state, greedy=False):
        if greedy or (np.random.random_sample() >= self.epsilon):
            action = np.argmax([self.Q.get(state, action) for action in range(self.nA)])
        else:
            action = np.random.choice(self.nA)
        return action

    def train(self, state, action, reward, next_state, done):
        best_next_action = self.choose_action(next_state, greedy=True)
        predicted_return = reward + self.gamma*self.Q.get(next_state, best_next_action)
        self.Q.update(state, action, predicted_return, self.alpha)

        if done:
            self.nEpisode += 1
            self.epsilon = 1.0/self.nEpisode
