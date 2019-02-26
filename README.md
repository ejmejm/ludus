# Ludus

Ludus is a reinforcement learning library for expediating development of RL applications and research. Ludus is still in an alpha development stage, so new algorithms and models are continuously being added with the growth of the framework. You can find many state-of-the-art algorithms implemented and ready to use. Additionly, the platform has ready to go integration with popular training environments like OpenAI's [gym](https://gym.openai.com/). The high level API of Ludus combined with easily accesible and well documented, open source code allows for both efficiency and control.

## Getting Started

These instructions will help you quickly get Ludus up and running, ready for RL application. Ludus is built on Python 3, and there is no gaurantee it will work with Python 2.

### Prerequisites

Ludus requires several libraries to get started. Anaconda 3 is recommended as it contains most of the required libraries, as well as many related libraries that may be useful. Nearly all prerequisites are installed when doing the standard pip installation as described below. The exeption is Tensorflow with GPU support. While __GPU enabled Tensorflow is strongly recommended__ over the CPU version, it is not required. You can find a guide to installing GPU enabled Tensorflow [here](https://www.tensorflow.org/install/gpu).

### Installing

The recommended method of installation is with the command, `pip install ludus`.

Alternatively, the package can be installed by cloning the repository and running, `pip setup.py install` in the root directory of the project.

## Your First Ludus Agent

To get started, the [vpg_cartpole example](https://github.com/ejmejm/ludus/blob/master/vpg_cartpole.ipynb) steps through the creation and training process for a simple agent. It is recommended that you use the notebook as an initial testing ground, and a template for other agents.

In Ludus, the process of creating an intelligent agent can be divided into 3 major steps. Performing the 1st step, and then repeating steps 2 and 3 in a training loop is the typical program flow:
1. __Creating input networks__
Depending on the type of trainer you wish to use to train your agent, a variety of different input neural networks may be required. The simplest form of this in Vanilla Policy Gradient (VPG / `VPGTrainer`), which requires only one network that maps observations to actions. Other, more complex methods like Proximal Policy Optimization (PPO / `PPOTrainer`) require two networks, one choosing actions and another estimating state values. If you are not familiar with these concepts, It is recommended that you thoroughly looking through the examples, as they are consice and easy to work with.

2. __Environment simulation & data gathering__
An `EnvController` instance is created and used to gather data from the environment. By adujsting the `n_threads` argument to an integer greater than 1, you can run multiple environments in parallel. The `sim_episodes` function is used to simualte the environment and gather data. Because Ludus handles environment simulation and data collection for you, custom environments can easily be integrated into the environment so long as they conform to a specific format (more on this [here](#using-a-custom-game-environment)). Once data has been gathered through an instance of `EnvController`, retrieving the data can be done with a `get_data()` call to the instance.

3. __Training__
 Training one epoch on the data is as simple as calling `network.train(ec.get_data())`, where network is an instance of a child of `BaseTrainer` (like `VPGTrainer`), and `ec` is an instance of `EnvController`. After a `get_data()` call, the training data memory buffer will be reset, unless otherwise in the function parameters.

## Using a Custom Game Environemnt

While this feature is supported and easy to implement, the documentation is not yet complete. For the time being, it is recommended that you examine OpenAI's [gym](https://gym.openai.com/). Creating an environment with the same `reset`, `step`, and initialization functions (the same input arguments and return values) will work with the Ludus framework.

## Built With

- [numpy](https://github.com/numpy/numpy) - Efficient mathematical operations
- [opencv2](https://github.com/opencv/opencv) - Image manipulation in 2D environments
- [Tensorflow](https://github.com/tensorflow/tensorflow) - Creating and training neural networks
- [gym](https://gym.openai.com/) - Envronments for training agents

## Documentation

Further, more in depth documentation is in the works, although not ready quite.

## Authors
- __Edan Meyer__ ([ejmejm](https://github.com/ejmejm)) is currently the lead and only developer for the project.

## License

This project is licensed under the MIT License. Please see the attached license file for more details.

