import functools
import os

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_hub as hub

# MUST import isaacgym before pytorch
import isaacgym
from improve.custom.algo.mlp.ppo import PPO, PPO_DEFAULT_CONFIG
from improve.custom.env.env_wrapper import wrap_env
from improve.custom.env.select_obs_wrapper import SelectObsWrapper
from improve.custom.memory.replay_buffer import ReplayBuffer
from improve.custom.models.mlp import MLPActorCriticPolicy
from improve.custom.trainers.sequential import SequentialTrainer
from skrl import config
from skrl.envs.loaders.torch.isaacgym_envs import \
    load_isaacgym_env_preview4 as make_env
from skrl.resources.preprocessors.jax.running_standard_scaler import \
    RunningStandardScaler
from skrl.utils import set_seed

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

if not hasattr(functools, "cache"):
    functools.cache = functools.lru_cache

jax.config.update("jax_enable_x64", False)


config.jax.backend = "jax"  # or "numpy"
set_seed(42)

cfg = PPO_DEFAULT_CONFIG.copy()
cfg["num_envs"] = 24
env = make_env(task_name="FrankaCubePickCamera",
               headless=True, num_envs=cfg["num_envs"])
env = wrap_env(env, wrapper='isaacgym-preview4')
env = SelectObsWrapper(env)
print(env.observation_space, env.action_space)

cfg["observation_space"] = env.observation_space
cfg["action_space"] = env.action_space
cfg["device"] = env.device

cfg["rollouts"] = 32
cfg["mini_batches"] = 8
cfg["learning_rate"] = 3e-5

cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {
    "size": env.observation_space
}

cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {
    "size": 1
}

cfg["kl_threshold"] = 0.008

# TODO: implement value clip, grad norm, lr scheduler,

memory = \
    ReplayBuffer(
        memory_size=cfg["rollouts"],
        num_envs=cfg["num_envs"]
    )

model = MLPActorCriticPolicy(
    obs_dim=env.observation_space.shape[0],
    act_dim=env.action_space.shape[0],
)

agent = \
    PPO(
        model=model,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=cfg["device"]
    )

cfg_trainer = {"timesteps": 1600000,
               "headless": False, "output_folder": "videos"}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
trainer.train()
