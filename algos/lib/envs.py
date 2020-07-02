import os
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import gym
from baselines import bench
import numpy as np
from baselines.common.vec_env import VecEnv
from baselines.common.vec_env.util import copy_obs_dict, dict_to_obs, obs_space_info


def make_thm_env(env_config, rank=0, log_dir=None, allow_early_resets=False):
    def _thunk():
        from algos.thm_env import TheoremProver
        env = TimeLimit(TheoremProver(env_config), env_config["time_limit"])

        if log_dir is not None:
            env = bench.Monitor(
                env,
                os.path.join(log_dir, str(rank)),
                allow_early_resets=allow_early_resets)

        return env

    return _thunk


def make_thm_vec_envs(env_config,
                      num_processes,
                      allow_early_resets=False,
                      log_dir=None,
                      ):
    envs = [
        make_thm_env(env_config, i, log_dir, allow_early_resets)
        for i in range(num_processes)
    ]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs, context='spawn')
    else:
        envs = DummyVecEnv(envs)

    return envs


class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        gym.Wrapper.__init__(self, env)
        self.max_episode_steps = max_episode_steps
        self.elapsed_steps = 0

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.elapsed_steps += 1
        if self.elapsed_steps > self.max_episode_steps:
            done = True
        return observation, reward, done, info

    def reset(self, index=None):
        self.elapsed_steps = 0
        return self.env.reset(index)


class DummyVecEnv(VecEnv):
    """
    VecEnv that does runs multiple environments sequentially, that is,
    the step and reset commands are send to one environment at a time.
    Useful when debugging and when num_env == 1 (in the latter case,
    avoids communication overhead)
    """
    def __init__(self, env_fns):
        """
        Arguments:

        env_fns: iterable of callables      functions that build environments
        """
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        obs_space = env.observation_space
        self.keys, shapes, dtypes = obs_space_info(obs_space)

        self.buf_obs = dict()
        self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool)
        self.buf_rews  = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        self.spec = self.envs[0].spec

    def step_async(self, actions):
        listify = True
        try:
            if len(actions) == self.num_envs:
                listify = False
        except TypeError:
            pass

        if not listify:
            self.actions = actions
        else:
            assert self.num_envs == 1, "actions {} is either not a list or has a wrong size - cannot match to {} environments".format(actions, self.num_envs)
            self.actions = [actions]

    def step_wait(self):
        for e in range(self.num_envs):
            action = self.actions[e]
            # if isinstance(self.envs[e].action_space, spaces.Discrete):
            #    action = int(action)

            obs, self.buf_rews[e], self.buf_dones[e], self.buf_infos[e] = self.envs[e].step(action)
            if self.buf_dones[e]:
                obs = self.envs[e].reset()
            self._save_obs(e, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones),
                self.buf_infos.copy())

    def reset(self):
        for e in range(self.num_envs):
            obs = self.envs[e].reset()
            self._save_obs(e, obs)
        return self._obs_from_buf()

    def _save_obs(self, e, obs):
        for k in self.keys:
            if k is None:
                self.buf_obs[k] = obs
            else:
                self.buf_obs[k] = obs[k]

    def _obs_from_buf(self):
        return [dict_to_obs(copy_obs_dict(self.buf_obs))]

    def get_images(self):
        return [env.render(mode='rgb_array') for env in self.envs]

    def render(self, mode='human'):
        if self.num_envs == 1:
            return self.envs[0].render(mode=mode)
        else:
            return super().render(mode=mode)