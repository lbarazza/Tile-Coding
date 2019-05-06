import sys
import gym
from agent import *

env = gym.make('Acrobot-v1')
env.seed(505);

tiling_specs = [(tuple(10 for _ in range(6)), (-0.08, -0.06, -0.04, -0.02, 0.9, 0.6)),
                (tuple(10 for _ in range(6)), (0.02, 0.0, -0.02, -0.04, -0.5, 0.6)),
                (tuple(10 for _ in range(6)), (-0.06, -0.04, 0.0, -0.06, -0.8, -0.6))]


agent = Agent((env.action_space.n, env.observation_space.low, env.observation_space.high), tiling_specs)



def run_agent(agent, nEpisodes, render=True):
    returns = []
    for i in range(nEpisodes):
        state = env.reset()
        ret = 0
        while True:
            action = agent.choose_action(state)
            new_state, reward, done, info = env.step(action)
            ret+=reward
            agent.train(state, action, reward, new_state, done)
            state = new_state

            if render and (i%100==0):
                env.render()

            if done:
                returns.append(ret)
                if (i >= 100) and (i%100==0):
                    print("Episode: {}/{}".format(i, nEpisodes), "Avg. return: ", np.mean(returns[-100:]))

                break


run_agent(agent, 15000)
