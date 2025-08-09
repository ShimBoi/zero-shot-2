import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import functools
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

from improve.custom.env.debug.debug_env_wrappers import *

if not hasattr(functools, "cache"):
    functools.cache = functools.lru_cache

jax.config.update("jax_enable_x64", False)

config.jax.backend = "jax"  # or "numpy"
set_seed(42)

cfg = PPO_DEFAULT_CONFIG.copy()
cfg["num_envs"] = 2048 # 2048
env = make_env(task_name="FrankaCubePickCamera",
               headless=True, num_envs=cfg["num_envs"])
env = wrap_env(env, wrapper='isaacgym-preview4')
env = SelectObsWrapper(env)

### DEBUG ENV WRAPPERS
# env = RandomObsOneRewardOneTimestepEnvWrapper(env)
# env = TwoObsTwoRewardOneTimestepEnvWrapper(env)
# env = TwoObsOneRewardTwoTimestepEnvWrapper(env)
# env = OneObsTwoActionTwoRewardOneTimestepEnvWrapper(env)
# env = TwoObsTwoActionTwoRewardOneTimestepEnvWrapper(env)
print(env.observation_space, env.action_space)

cfg["observation_space"] = env.observation_space
cfg["action_space"] = env.action_space
cfg["device"] = env.device

cfg["normalize_advantage"] = True
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95

# for not have static lr (need Linear decay?)
cfg["learning_rate"] = 5e-4
cfg["learning_epochs"] = 5

cfg["kl_threshold"] = 0.008

cfg["entropy_loss_scale"] = 0.0

cfg["rollouts"] = 128
cfg["mini_batches"] = 4096
cfg["learning_rate"] = 3e-5

cfg["grad_norm_clip"] = 1
cfg["value_loss_scale"] = 4

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
               "headless": False, "output_folder": "videos", "save_video": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
trainer.train()
