# PCGRL OpenAI GYM Interface
An [OpenAI GYM](https://gym.openai.com/) environment for Procedural Content Generation via Reinforcement Learning (PCG-RL).

The framework, along with some initial reinforcement learning results, is covered in the paper [Procedural Content Generation via Reinforcement Learning](). This paper should be cited if code from this project is used in any way:
```
@inproceedings{khalifa2020pcgrl,
  title={PCGL-RL: Procedural Content Generation via Reinforcement Learning},
  author={Khalifa, Ahmed and [Ruben]? and Bontrager, Philip and Togelius, Julian},
  booktitle={},
  year={2020},
  organization={}
}
```

## Installation
1. Clone this repo to your local machine.
2. To install the package, run `pip install -e .` from inside the repo folder. (Don't worry it will install OpenAI GYM environment automatically, otherwise you can install it first by following that [link](https://github.com/openai/gym#installation))
3. If everything went fine, the PCG-RL gym interface is ready to be used. Check the [following section](https://github.com/amidos2006/gym-pcgrl#usage) on how to use it.

## Usage
The PCG-RL GYM interface have multiple different environments where each environment consists of two parts a problem and a representation. All the environment follow the following name convention:
```
[problem_name]-[representation_name]-[version]
```
For the full list of supported problems names check the [Supported Problems](https://github.com/amidos2006/gym-pcgrl#supported-problems) section and for the full list of the supported representations name check the [Supported Representations](https://github.com/amidos2006/gym-pcgrl#supported-representations) section.

To list all the registered environments, you can run the following code:
```python
[env.id for env in envs.registry.all() if "gym_pcgrl" in env.entry_point]
```

After installing the interface, you can use it like any other GYM interface. Here is a simple example on how to use the framework on the Sokoban environment with Narrow representation:

```python
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

## Supported Problems
Problems are the current games that we want to apply PCG-RL towards them. The following table lists all the supported problems in the interface:

| Name       | Goal                                                                                                                                                                        | Tile Values                                                                                                                                                |
|------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| binary     | Generate a fully connected top down layout where the longest path is greater than a certain threshold                                                                       | 0: empty, 1: solid                                                                                                                                         |
| dungeon    | Generate a fully connected level for a simple dungeon crawler similar to [MiniDungeons 1](http://minidungeons.com/) where the player has to kill 50% of enemies before done | 0: empty, 1: solid, 2: player (max of 5 health), 3: exit, 4: potion (restores 2 health), 5: treasure, 6: goblin (deals 1 damage), 7: ogre (deals 2 damage) |
| platformer | Generate a fully connected level for a simple platformer similar to [Dangerous Dave](http://www.dangerousdave.com) where the player has to jump at least 2 times to finish  | 0: empty, 1: solid, 2: player, 3: exit, 4: diamonds, 5: spikes                                                                                             |
| sokoban    | Generate a fully connected [Sokoban](https://en.wikipedia.org/wiki/Sokoban) level that can be solved                                                                        | 0: empty, 1: solid, 2: player, 3: crate (to be pushed toward the target), 4: target (the location where the crate should ends)                             |

## Supported Representations
Representations are the way the Procedural Content Generation problem is formatted as a [Markov Decision Process](https://en.wikipedia.org/wiki/Markov_decision_process) to be able to use it for reinforcement learning. All the problems can be represented using any of the supported representations. The following table shows all the supported representations in the interface:

| Name   | Observation Space                                                                                  | Action Space                                                                                                                             |
|--------|----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| narrow | 2D Box of integers that represent the map and 1D Box of integers that represents the x, y position | One Discrete space that represents the new tile value                                                                                    |
| wide   | 2D Box of integers that represent the map                                                          | Three Discrete spaces that represent the x position, y position, new tile value                                                          |
| turtle | 2D Box of integers that represent the map and 1D Box of integers that represents the x, y position | One Discrete space where the first 4 actions move the turtle (left, right, up, or down) while the rest of actions are for the tile value |

The `narrow`, `wide`, and `turtle` representation are adapted from [Tree Search vs Optimization Approaches for Map Generation](https://arxiv.org/pdf/1903.11678.pdf) work by Bhaumik et al.

## Create your own environment
to be written

## Create your own representation
to be written

## Contributing
Bug reports and pull requests are welcome on GitHub at [https://github.com/amidos2006/gym-pcgrl](https://github.com/amidos2006/gym-pcgrl).

## License
This code is available as open source under the terms of the [MIT License](https://opensource.org/licenses/MIT).
