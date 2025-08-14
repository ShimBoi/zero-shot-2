import isaacgym

import flax.linen as nn
import jax
import jax.numpy as jnp

# import the skrl components to build the RL system
from skrl import config
from skrl.envs.loaders.jax import load_isaacgym_env_preview4
from improve.custom.env.env_wrapper import wrap_env
from skrl.models.jax import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.jax.running_standard_scaler import RunningStandardScaler
from skrl.resources.schedulers.jax import KLAdaptiveRL
from skrl.utils import set_seed

from improve.custom.env.select_obs_wrapper import CustomIsaacGymEnvWrapper
from improve.custom.trainers.sequential import SequentialTrainer
from improve.custom.memory.replay_buffer import ReplayBuffer
from improve.custom.algo.mlp.ppo import PPO, PPO_DEFAULT_CONFIG
from improve.custom.mixins.a2c_mixin import A2CMixin

config.jax.backend = "jax"  # or "numpy"

# seed for reproducibility
set_seed(42)  # e.g. `set_seed(42)` for fixed seed

class A2CModel(A2CMixin, Model):
    def __init__(self, observation_space, action_space, device=None, clip_actions=False,
                 clip_log_std=False, min_log_std=-20, max_log_std=2, reduction="sum", clip_values=False, **kwargs):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        A2CMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction, clip_values)

    @nn.compact
    def __call__(self, inputs, role):
        x = nn.elu(nn.Dense(256)(inputs["states"]))
        x = nn.elu(nn.Dense(128)(x))
        x = nn.elu(nn.Dense(64)(x))
        mean_actions = nn.Dense(self.num_actions)(x)
        log_std = self.param("log_std", lambda _: jnp.ones(self.num_actions))
        value = nn.Dense(1)(x)

        return mean_actions, log_std, value, {}
    

device = "cuda" if jax.devices("gpu") else "cpu"
print(f"Using device: {device}")

cfg = PPO_DEFAULT_CONFIG.copy()
# CUSTOM CONFIGS
cfg["timesteps"] = 1600000
cfg["num_envs"] = 8192
cfg["save_video"] = True
cfg["experiment"]["wandb"] = True 

        
# load and wrap the Isaac Gym environment
env = load_isaacgym_env_preview4(task_name="FrankaCubePick", headless=True, num_envs=cfg["num_envs"])
env = wrap_env(env)
env = CustomIsaacGymEnvWrapper(env)

cfg["rollouts"] = 96  # memory_size
cfg["learning_epochs"] = 5
cfg["mini_batches"] = 4  # 96 * 4096 / 98304
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
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 336
cfg["experiment"]["checkpoint_interval"] = 3360
cfg["experiment"]["directory"] = "runs/jax/Isaac-Lift-Franka-v0"

# instantiate a memory as rollout buffer (any memory can be used for this)
memory = ReplayBuffer(memory_size=cfg["rollouts"], num_envs=env.num_envs)

# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
models = {}
models["a2c"] = A2CModel(
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device
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