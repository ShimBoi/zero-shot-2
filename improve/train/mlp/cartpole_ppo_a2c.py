import isaacgym

import flax.linen as nn
import jax
import jax.numpy as jnp

# import the skrl components to build the RL system
from skrl import config
from skrl.envs.loaders.jax import load_isaacgym_env_preview4
from improve.custom.env.env_wrapper import wrap_env
from skrl.models.jax import DeterministicMixin, GaussianMixin, Model
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
        x = nn.elu(nn.Dense(32)(inputs["states"]))
        x = nn.elu(nn.Dense(32)(x))
        mean_actions = nn.Dense(self.num_actions)(x)
        log_std = self.param("log_std", lambda _: jnp.zeros(self.num_actions))
        value = nn.Dense(1)(x)

        return mean_actions, log_std, value, {}
        
# load and wrap the Isaac Gym environment
env = load_isaacgym_env_preview4(task_name="Cartpole", headless=True)
env = wrap_env(env)
env = CustomIsaacGymEnvWrapper(env)

device = env.device


# instantiate a memory as rollout buffer (any memory can be used for this)
memory = ReplayBuffer(memory_size=16, num_envs=env.num_envs)

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


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 16  # memory_size
cfg["learning_epochs"] = 8
cfg["mini_batches"] = 1  # 16 * 512 / 8192
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 3e-4
# cfg["learning_rate_scheduler"] = KLAdaptiveRL
# cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["grad_norm_clip"] = 1.0
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True
cfg["entropy_loss_scale"] = 0.0
cfg["value_loss_scale"] = 2.0
cfg["kl_threshold"] = 0
cfg["rewards_shaper"] = lambda rewards, timestep, timesteps: rewards * 0.1
# cfg["state_preprocessor"] = RunningStandardScaler
# cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
# cfg["value_preprocessor"] = RunningStandardScaler
# cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 16
cfg["experiment"]["checkpoint_interval"] = 80
cfg["experiment"]["directory"] = "runs/jax/Cartpole"
cfg["experiment"]["wandb"] = False
cfg["save_video"] = True

agent = PPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 16000, "headless": True, "save_video": cfg["save_video"], "output_folder": "videos"}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
trainer.train()