from gym import Wrapper, spaces
import random
import numpy as np
import gym
from gym.envs.registration import register
from typing import List, Any

register(id = "Trading-v0", entry_point = "storm.environment.wrapper:EnvironmentWrapper")

class EnvironmentWrapper(Wrapper):
    def __init__(self,
                 env: Any,
                 transition_shape = None,
                 seed=42):
        super().__init__(env)
        self.seed = seed

        self.env = env

        random.seed(seed)
        np.random.seed(seed)

        self.actions = env.actions

        action_shape = transition_shape["actions"]["shape"]
        action_type = transition_shape["actions"]["type"]
        state_shape = transition_shape["states"]["shape"][1:]
        state_type = transition_shape["states"]["type"]
        print("action shape {}, action type {}, state shape {}, state type {}".format(action_shape, action_type,
                                                                                      state_shape, state_type))

        self.action_space = spaces.Discrete(
            n = env.action_dim,
        )
        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=state_shape,
            dtype=state_type,
        )

        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def reset(self):
        state, info = self.env.reset()
        return state, info

    def step(self, action):
        next_state, reward, done, truncted, info = self.env.step(action)
        return next_state, reward, done, truncted, info


def make_env(env_id, env_params):
    def thunk():
        env = gym.make(env_id, **env_params)
        return env
    return thunk