# PCGRL OpenAI GYM Interface
An [OpenAI GYM](https://gym.openai.com/) environment for Procedural Content Generation via Reinforcement Learning (PCGRL).

The framework, along with some initial reinforcement learning results, is covered in the paper [Procedural Content Generation via Reinforcement Learning](). This paper should be cited if code from this project is used in any way:
```
@inproceedings{khalifa2020pcgrl,
  title={PCGRL: Procedural Content Generation via Reinforcement Learning},
  author={Khalifa, Ahmed and Bontrager, Philip and Earle, Sam and Togelius, Julian},
  booktitle={},
  year={2020},
  organization={}
}
```

## Installation
1. Clone this repo to your local machine.
2. To install the package, run `pip install -e .` from inside the repo folder. (Don't worry it will install OpenAI GYM environment automatically, otherwise you can install it first by following that [link](https://github.com/openai/gym#installation))
3. If everything went fine, the PCGRL gym interface is ready to be used. Check the [following section](https://github.com/amidos2006/gym-pcgrl#usage) on how to use it.

## Usage
The PCGRL GYM interface have multiple different environments where each environment consists of two parts a problem and a representation. All the environment follow the following name convention:
```
[problem_name]-[representation_name]-[version]
```
For the full list of supported problems names check the [Supported Problems](https://github.com/amidos2006/gym-pcgrl#supported-problems) section and for the full list of the supported representations name check the [Supported Representations](https://github.com/amidos2006/gym-pcgrl#supported-representations) section.

To list all the registered environments, you can run the following code:
```python
from gym import envs
import gym_pcgrl

[env.id for env in envs.registry.all() if "gym_pcgrl" in env.entry_point]
```

After installing the interface, you can use it like any other GYM interface. Here is a simple example on how to use the framework on the Sokoban environment with Narrow representation:

```python
import gym
import gym_pcgrl

env = gym.make('sokoban-narrow-v0')
obs = env.reset()
for t in range(1000):
  action = env.action_space.sample()
  obs, reward, done, info = env.step(action)
  env.render('human')
  if done:
    print("Episode finished after {} timesteps".format(t+1))
    break
```

Beside the OpenAI GYM traditional functions. Our interface supports additional functionalities such as:
- `self.get_num_tiles()`: This function get the number of different tiles that can appear in the observation space
- `get_border_tile()`: This function get the tile index to be used for padding a certain problem. It is used by certain wrappers.
- `adjust_param(**kwargs)`: This function that helps adjust the problem and/or representation parameters such as modifying `width` and `height` of the generated map.

## Supported Problems
Problems are the current games that we want to apply PCGRL towards them. The following table lists all the supported problems in the interface:

| Name     | Goal                                                                                                                                                                        | Tile Values                                                                                                                                                |
|----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| binary   | Generate a fully connected top down layout where the longest path is greater than a certain threshold                                                                       | 0: empty, 1: solid                                                                                                                                         |
| ddave    | Generate a fully connected level for a simple platformer similar to [Dangerous Dave](http://www.dangerousdave.com) where the player has to jump at least 2 times to finish  | 0: empty, 1: solid, 2: player, 3: exit, 4: diamonds, 5: trophy (act like a key for the exit), 6: spikes                                                    |
| mdungeon | Generate a fully connected level for a simple dungeon crawler similar to [MiniDungeons 1](http://minidungeons.com/) where the player has to kill 50% of enemies before done | 0: empty, 1: solid, 2: player (max of 5 health), 3: exit, 4: potion (restores 2 health), 5: treasure, 6: goblin (deals 1 damage), 7: ogre (deals 2 damage) |
| sokoban  | Generate a fully connected [Sokoban](https://en.wikipedia.org/wiki/Sokoban) level that can be solved                                                                        | 0: empty, 1: solid, 2: player, 3: crate (to be pushed toward the target), 4: target (the location where the crate should ends)                             |

## Supported Representations
Representations are the way the Procedural Content Generation problem is formatted as a [Markov Decision Process](https://en.wikipedia.org/wiki/Markov_decision_process) to be able to use it for reinforcement learning. All the problems can be represented using any of the supported representations. The following table shows all the supported representations in the interface:

| Name   | Observation Space                                                                                  | Action Space                                                                                                                             |
|--------|----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| narrow | 2D Box of integers that represent the map and 1D Box of integers that represents the x, y position | One Discrete space that represents the new tile value                                                                                    |
| wide   | 2D Box of integers that represent the map                                     | Three Discrete spaces that represent the x position, y position, new tile value                                                          |
| turtle | 2D Box of integers that represent the map and 1D Box of integers that represents the x, y position | One Discrete space where the first 4 actions move the turtle (left, right, up, or down) while the rest of actions are for the tile value |

The `narrow`, `wide`, and `turtle` representation are adapted from [Tree Search vs Optimization Approaches for Map Generation](https://arxiv.org/pdf/1903.11678.pdf) work by Bhaumik et al.

## Create your own problem
Create the new problem class in the `gym_pcgrl.envs.probs` and extends `Problem` class from `gym_pcgrl.envs.probs.problem`. This class has to implement the following functions.
```python
def __init__(self):
  super().__init__()
  ...

def get_tile_types(self):
  ...

def get_stats(self, map):
  ...

def get_reward(self, new_stats, old_stats):
  ...

def get_episode_over(self, new_stats, old_stats):
  ...

def get_debug_info(self, new_stats, old_stats):
  ...
```
Also, you need to make sure that you setup the following parameters in the constructor:
- `self._width`: the generated map width.
- `self._height`: the generated map height.
- `self._prob`: a dictionary for all the game tiles where keys are the tile names and the values are the probability of the tile appearing when initializing a random map.
- `self._border_size`: the size of the border added around the generated level (in a lot of games there might be a border surrounding the level, it is a good idea to get that out).
- `self._border_tile`: the tile name used for the border.
- `self._tile_size`: the size of the tile in pixels to be used in rendering.
- `self._graphics`: a dictionary for all the game graphics where keys are the tile names and values are the Pillow images for rendering the problem.

Feel free to override any other function if you need a behavior different from the normal behavior. For example: In all our problems, we want our system to not load the graphics unless it is going to render it. We override `render()` function so we can initialize `self._graphics` at the beginning of the `render()` instead of the constructor.

After implementing your own class, you need to add the name and the class in `gym_pcgrl.envs.probs.PROBLEMS` dictionary that can be found in [__init__.py](https://github.com/amidos2006/gym-pcgrl/blob/master/gym_pcgrl/envs/probs/__init__.py) the key name is used as the problem name for the environment and the value is to refer to the main class that it need to construct for that problem.

## Create your own representation
Create the new representation class in the `gym_pcgrl.envs.reps` and extends `Representation` class from `gym_pcgrl.envs.reps.representation`. This class has to implement the following functions.
```python
def __init__(self, width, height, prob):
  super().__init__(width, height, prob)
  ...

def get_action_space(self):
  ...

def get_observation_space(self):
  ...

def get_observation(self):
  ...

def update(self, action):
  ...
  # boolean to define where the change happened and x,y for the location of change if it happened
  return change, x, y
```
Feel free to override any other function if you need a behavior different from the normal behavior. For example: in the `narrow` representation, we wanted to show the location where the agent should change on the rendered image. We override the `render()` function to draw a red square around the correct tile.

After implementing your own class, you need to add the name and the class in `gym_pcgrl.envs.reps.REPRESENTATIONS` dictionary that can be found in [__init__.py](https://github.com/amidos2006/gym-pcgrl/blob/master/gym_pcgrl/envs/reps/__init__.py) the key name is used as the representation name for the environment and the value is to refer to the main class that it need to construct for that representation.

## Contributing
Bug reports and pull requests are welcome on GitHub at [https://github.com/amidos2006/gym-pcgrl](https://github.com/amidos2006/gym-pcgrl).

## License
This code is available as open source under the terms of the [MIT License](https://opensource.org/licenses/MIT).
