# Tile Coding
Implementation of the Tile Coding algorithm

In Reinforcement Learning Tile Coding is an approach to deal with continous state space problems while using discrete state space algorithms. This algorithm adds to the normal discretization of space technique, which breaks up the state space into discrete pieces forming a grid which is the Q-table. Tile Coding creates many grids, each with a different offset and size. Each point in the continous space is then located by a list of the indexes of the cells that the point falls into for each grid. By doing this, every time you update the new Q-table at a point, you're updating many cells in different grids which then yields to a more smooth representation of the state space.

The implementation in this repository uses tile coding to create a Q-table, which is then managed by a Q-learning agent in the Acrobot-v1 environment from OpenAI's Gym.

TODO:
  - implement Adaptive Tile Coding, which automatically splits cells into multiple cells where needed to learn better
