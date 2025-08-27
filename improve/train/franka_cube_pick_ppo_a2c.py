from improve.custom.models.rt1.rt1 import RT1
from improve.custom.models.rt1.rtx_wrapper import RTXPPO
import isaacgym

import flax.linen as nn
import jax
import jax.numpy as jnp

# import the skrl components to build the RL system
from skrl import config
from skrl.envs.loaders.jax import load_isaacgym_env_preview4
from improve.custom.env.env_wrapper import wrap_env
from skrl.models.jax import Model
from skrl.resources.preprocessors.jax.running_standard_scaler import RunningStandardScaler
from skrl.resources.schedulers.jax import KLAdaptiveRL
from skrl.utils import set_seed

from improve.custom.env.select_obs_wrapper import CustomIsaacGymEnvWrapper
from improve.custom.trainers.sequential import SequentialTrainer
from improve.custom.memory.replay_buffer import ReplayBuffer
from improve.custom.algo.rtx.ppo import PPO, PPO_DEFAULT_CONFIG
from improve.custom.mixins.a2c_mixin import A2CMixin

config.jax.backend = "jax"  # or "numpy"
jax.config.update('jax_enable_x64', False)  # Use 32-bit instead of 64-bit
jax.config.update('jax_platform_name', 'gpu')

# seed for reproducibility
set_seed(42)  # e.g. `set_seed(42)` for fixed seed

device = "cuda" if jax.devices("gpu") else "cpu"
print(f"Using device: {device}")

cfg = PPO_DEFAULT_CONFIG.copy()
# CUSTOM CONFIGS
cfg["timesteps"] = 1600000
cfg["num_envs"] = 32
cfg["save_video"] = True
cfg["experiment"]["wandb"] = False
cfg["is_image_obs"] = True
        
# load and wrap the Isaac Gym environment
env = load_isaacgym_env_preview4(task_name="FrankaCubePickCamera", headless=True, num_envs=cfg["num_envs"])
env = wrap_env(env)
env = CustomIsaacGymEnvWrapper(env, env.observation_space.shape)
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")

cfg["rollouts"] = 96  # memory_size
cfg["learning_epochs"] = 5
cfg["mini_batches"] = 96 * 8 # 96 * 4096 / 98304
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 1e-3
cfg["learning_rate_scheduler"] = KLAdaptiveRL
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.01, "min_lr": 1e-5}
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["grad_norm_clip"] = 1.0
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True
cfg["entropy_loss_scale"] = 0.01
cfg["value_loss_scale"] = 1.0
cfg["kl_threshold"] = 0
cfg["rewards_shaper"] = None
cfg["time_limit_bootstrap"] = True
# cfg["state_preprocessor"] = RunningStandardScaler
# cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
# cfg["value_preprocessor"] = RunningStandardScaler
# cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 336
cfg["experiment"]["checkpoint_interval"] = 3360
cfg["experiment"]["directory"] = "runs/jax/Isaac-Lift-Franka-Image"

# instantiate a memory as rollout buffer (any memory can be used for this)
memory = ReplayBuffer(memory_size=cfg["rollouts"], num_envs=env.num_envs)

# instantiate the agent's models (function approximators).
models = {}
models["a2c"] = RTXPPO(
    env.observation_space, 
    env.action_space, 
    device, 
    task="Pick up the red cube",
    clip_actions=True,
    clip_values=False,
    reduction="sum",
)

# instantiate models' state dict
for role, model in models.items():
    model.init_state_dict(role)


agent = PPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": cfg["timesteps"], "headless": True, "save_video": cfg["save_video"], "output_folder": "videos"}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
trainer.train()