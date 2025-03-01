import isaacgym
import os
import flax.linen as nn
import jax.numpy as jnp
from skrl import config
from improve.custom.algo.ppo import PPO, PPO_DEFAULT_CONFIG
from improve.custom.env.env_wrapper import wrap_env
from skrl.memories.jax import RandomMemory
from skrl.models.jax import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.jax import RunningStandardScaler
from skrl.resources.schedulers.jax import KLAdaptiveRL
from improve.custom.trainers.sequential import SequentialTrainer
from skrl.utils import set_seed
from skrl.utils.huggingface import download_model_from_huggingface

from improve.custom.env.loader import make_env
from improve.custom.models.rt1 import RT1
from improve.custom.models.rt1_policy import RT1Policy

config.jax.backend = "jax"  # or "numpy"


# seed for reproducibility
set_seed(42)  # e.g. `set_seed(42)` for fixed seed

# define models (stochastic and deterministic models) using mixins
sequence_length = 15
num_action_tokens = 11
layer_size = 256
vocab_size = 512
num_image_tokens = 81
rt1x_model = RT1(
    num_image_tokens=num_image_tokens,
    num_action_tokens=num_action_tokens,
    layer_size=layer_size,
    vocab_size=vocab_size,
    # Use token learner to reduce tokens per image to 81.
    use_token_learner=True,
    # RT-1-X uses (-2.0, 2.0) instead of (-1.0, 1.0).
    world_vector_range=(-2.0, 2.0),
)
policy = RT1Policy(
    checkpoint_path='rt_1_x_jax',
    model=rt1x_model,
    seqlen=sequence_length,
)
print("DONE")
quit()


class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device=None, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum", **kwargs):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        GaussianMixin.__init__(
            self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

    @nn.compact  # marks the given module method allowing inlined submodules
    def __call__(self, inputs, role):
        print(inputs)
        x = nn.elu(nn.Dense(32)(inputs["states"]))
        x = nn.elu(nn.Dense(32)(x))
        x = nn.Dense(self.num_actions)(x)
        log_std = self.param("log_std", lambda _: jnp.zeros(self.num_actions))
        return x, log_std, {}


class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device=None, clip_actions=False, **kwargs):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        DeterministicMixin.__init__(self, clip_actions)

    @nn.compact  # marks the given module method allowing inlined submodules
    def __call__(self, inputs, role):
        x = nn.elu(nn.Dense(32)(inputs["states"]))
        x = nn.elu(nn.Dense(32)(x))
        x = nn.Dense(1)(x)
        return x, {}


# load and wrap the Isaac Gym environment
env = make_env(task_name="Cartpole", headless=False)
env = wrap_env(env, wrapper='isaacgym-preview4')

device = env.device


# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=16, num_envs=env.num_envs, device=device)


# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
models = {}
models["policy"] = Policy(env.observation_space, env.action_space, device)
models["value"] = Value(env.observation_space, env.action_space, device)

# instantiate models' state dict
for role, model in models.items():
    model.init_state_dict(role)

cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 16  # memory_size
cfg["learning_epochs"] = 8
cfg["mini_batches"] = 1  # 16 * 512 / 8192
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 3e-4
cfg["learning_rate_scheduler"] = KLAdaptiveRL
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
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
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {
    "size": env.observation_space, "device": device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

cfg["experiment"] = {
    "write_interval": 16,
    "checkpoint_interval": 80,
    "directory": "./runs/jax/Cartpole",
    "store_separately": False,
    "wandb": False,
    "wandb_kwargs": {
        "project": "skrl",
        "entity": "shimboi",
        "name": "PPO-Cartpole"
    }
}

agent = PPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 1600, "headless": False}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

path = download_model_from_huggingface(
    "skrl/IsaacGymEnvs-Cartpole-PPO", filename="agent.pickle")
agent.load(path)

# start training
trainer.train()


# start evaluation
# trainer.eval()
