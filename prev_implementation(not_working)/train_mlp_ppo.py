import os

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import functools
import jax
import argparse
import os

# MUST import isaacgym before pytorch
import isaacgym
from improve.custom.algo.mlp.ppo import PPO, PPO_DEFAULT_CONFIG
from improve.custom.env.env_wrapper import wrap_env
from improve.custom.env.select_obs_wrapper import CustomIsaacGymEnvWrapper
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
cfg["num_envs"] = 8192 # 2048
env = make_env(task_name="FrankaCubePick",
               headless=True, num_envs=cfg["num_envs"])
env = wrap_env(env, wrapper='isaacgym-preview4')
env = CustomIsaacGymEnvWrapper(env)
print(env.observation_space, env.action_space)


cfg["observation_space"] = env.observation_space
cfg["action_space"] = env.action_space
cfg["device"] = env.device

cfg["normalize_advantage"] = True
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95

# for not have static lr (need Linear decay?)
cfg["learning_rate_scheduler"] = "linear"
cfg["learning_rate"] = 5e-4
cfg["learning_epochs"] = 5

cfg["kl_threshold"] = 0.008

cfg["entropy_loss_scale"] = 0.0

cfg["timesteps"] = 1600000
cfg["rollouts"] = 32
cfg["mini_batches"] = 16

cfg["grad_norm_clip"] = 1
cfg["value_loss_scale"] = 1

cfg["time_limit_bootstrap"] = True
cfg["clip_predicted_values"] = True
cfg["value_clip"] = 0.2

# cfg["state_preprocessor"] = RunningStandardScaler
# cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": cfg["device"]}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = None

cfg["experiment"]["write_interval"] = 100
cfg["experiment"]["wandb"] = True
cfg["save_video"] = True

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

cfg_trainer = {"timesteps": cfg["timesteps"],
               "headless": False, "output_folder": "videos", "save_video": cfg["save_video"]}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
trainer.train()
