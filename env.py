import numpy as np
import threading
import time
import gym
from ludus.cart_pole_cont import CartPoleEnv
from ludus.memory import MTMemoryBuffer

def make_cart_pole():
    """Make and return a discrete cart pole environment from OpenAI gym"""
    return gym.make('CartPole-v1')

def make_cart_pole_c():
    """Make and return a continuous cart pole environment adapted from OpenAI gym"""
    return CartPoleEnv()

def make_car_race():
    """Make and return a continuous cart car race environment from OpenAI gym"""
    return gym.make('CarRacing-v0')

def make_lunar_lander_d():
    """Make and return a discrete lunar lander environment from OpenAI gym"""
    return gym.make('LunarLander-v2')

def make_lunar_lander_c():
    """Make and return a continuous lunar lander environment from OpenAI gym"""
    return gym.make('LunarLanderContinuous-v2')

class EnvController():
    """The EnvController is used to keep track of the environment being used,
    and also the game loop.
    
    It implements Python's multithreading module to run several game loops in 
    parallel and efficiently generate data.
    """
    def __init__(self, make_env, n_threads=1, memory_buffer=MTMemoryBuffer(), 
                 obs_transform=None, act_transform=None):
        """Initialization for the EnvController simple involved setting variables
        and creating locks to prepare for parallel environment simulation.

        Args:
            make_env: Function that creates and returns a new instance of the desired environment.
            n_threads (int, optional): Number of threads for env simulation.
            memory_buffer: An instance of a multi-threaded memory buffer to store episode data.
            obs_transform: Function that is applied to all observations before use.
            act_transform: Function that is applied to all actions before use.
        """
        self.make_env = make_env
        self.mb = memory_buffer
        self.n_threads = n_threads
        if obs_transform is not None:
            self.obs_transform = obs_transform
        if act_transform is not None:
            self.act_transform = act_transform
        self.act_lock = threading.Lock()
        self.init_lock = threading.Lock()
        
    def obs_transform(self, obs):
        """Uses the stored observation transformation function to transform the given observation.
        
        Args:
            obs: Observation from the environment.
        
        Returns:
            Observation after formated by the instance's obs transformation function.
        """
        return obs.squeeze()
    
    def act_transform(self, act):
        """Uses the stored action transformation function to transform the given action.
        
        Args:
            act: Action chosen by the agent.
        
        Returns:
            Action after formated by the instance's act transformation function.
        """
        return act
    
    def set_obs_transform(self, transform_func):
        """Sets the observation transformation function.
        
        Args:
            transform_func: Function that is applied to every observation from the environment 
                before the observations are recorded and used.
        """
        self.obs_transform = transform_func
    
    def set_act_transform(self, transform_func):
        """Sets the action transformation function.
        
        Args:
            transform_func: Function that is applied to every action from the environment 
                before the actions are recorded and used.
        """
        self.act_transform = transform_func
        
    def sim_thread(self, agent_id, network, n_episodes=1, max_steps=200, render=False):
        """Playthrough episodes of the instance's environment, collecting data in the memory
        buffer that can be used for training or other purposes. The data from the memory buffer
        is not extracted until a call to the 'get_data' method.
        
        Args:
            agent_id (int): ID of the thread running these episodes, used in the memory 
                buffer. Should be unique from any other agent IDs being used by any other 
                workers running in parallel. Failure to keep the ID unique for the timeframe 
                will result in incorrect discounted rewards in the final data, as well as
                out of order data.
            network: Trainer that extends ludus.policies.BaseTrainer. The network is used
                to generate actions at each step in the environment.
            n_episodes (int): The number of rollouts to generate.
            max_steps (int): Max number of steps that should be taken in a single rollout.
                Generally equivelant to the max number of frames before cutting off an episode.
            render (boolean): True to render the playthroughs, False to run with no UI. All
                traing should be done without rendering for significantly faster speeds.
                Rendering for the purpose of tracking progress or curiosity should ideally be
                done with the 'render_episodes' method.
        """
        with self.init_lock:
            env = self.make_env()
        
        for episode in range(n_episodes):
            self.mb.start_rollout(agent_id)
            obs = env.reset()
            obs = self.obs_transform(obs)
            for step in range(max_steps):
                with self.act_lock:
                    act = network.gen_act(obs)
                act = self.act_transform(act)

                obs_next, rew, d, _ = env.step(act)
                obs_next = self.obs_transform(obs_next)

                if render:
                    env.render()
                    time.sleep(0.02)

                self.mb.record(agent_id, obs, act, rew, obs_next)
                obs = obs_next

                if d:
                    break
                    
    def sim_episodes(self, network, n_episodes=1, max_steps=200, render=False, return_data=False):
        threads = []
        ept = [int(n_episodes // self.n_threads) for i in range(self.n_threads)] # Episodes per thread
        ept[:(n_episodes % self.n_threads)] += np.ones((n_episodes % self.n_threads,))
        for i in range(self.n_threads):
            new_thread = threading.Thread(target=self.sim_thread, args=(i, network, int(ept[i]), max_steps, render))
            threads.append(new_thread)
            new_thread.start()
            
        for thread in threads:
            thread.join()
        
        if return_data:
            return self.mb.to_data()
        
    def get_avg_reward(self):
        return self.mb.get_avg_reward()
    
    def get_data(self):
        return self.mb.to_data()
    
    def render_episodes(self, network, n_episodes=1, max_steps=200):
        with self.init_lock:
            env = self.make_env()
        
        for episode in range(n_episodes):
            obs = env.reset()
            obs = self.obs_transform(obs)
            for step in range(max_steps):
                with self.act_lock:
                    act = network.gen_act(obs)
                act = self.act_transform(act)

                obs_next, rew, d, _ = env.step(act)
                obs_next = self.obs_transform(obs_next)
                obs = obs_next

                env.render()
                time.sleep(0.02)

                if d:
                    break
        
