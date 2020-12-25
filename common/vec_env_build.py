import gym
from vecEnv import DummyVecEnv, SubprocVecEnv
from common import wrapEnv


def makeVecEnv(env_name, num_env, seed, start_method):
    def make_env(rank):
        def _thunk():
            env = gym.make(env_name)
            env.seed(seed + rank)
            return env

        return _thunk

    if num_env == 1:
        return DummyVecEnv([make_env(i) for i in range(num_env)])
    return SubprocVecEnv(
        [make_env(i) for i in range(num_env)], start_method=start_method
    )


def makeAtariVecEnv(env_name, num_env, seed, start_method):
    """Vectorized environments
  If more than 1 environment, extra processes are triggered. Each process has its own environment.
  The seed of each env has to be different so that each environment is unique.
  Note: VecEnv automatically call env.reset() if it is done. The last terminate obs will not be 
  returned directly. But can be manually returned by configuring params. The next start obs is returned.
  """

    def make_env(rank):
        def _thunk():
            env = gym.make(env_name)
            env.seed(seed + rank)
            return wrapEnv(env)

        return _thunk

    if num_env == 1:
        return DummyVecEnv([make_env(i) for i in range(num_env)])
    return SubprocVecEnv(
        [make_env(i) for i in range(num_env)], start_method=start_method
    )

