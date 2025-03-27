import isaacgym
import os
import flax.linen as nn
import jax.numpy as jnp
from skrl import config
from improve.custom.algo.ppo import PPO, PPO_DEFAULT_CONFIG
from improve.custom.env.env_wrapper import wrap_env, ImageObservationWrapper
from skrl.memories.jax import RandomMemory
from skrl.resources.schedulers.jax import KLAdaptiveRL
from improve.custom.trainers.sequential import SequentialTrainer
from skrl.utils import set_seed

from improve.custom.env.loader import make_env
from improve.custom.models.rt1 import RT1
from improve.custom.models.rt1_policy import RT1Policy
from improve.custom.models.rt1_critic import RT1Critic
from improve.custom.preprocessors.rtx_preprocessor import RTXPreprocessor

config.jax.backend = "jax"  # or "numpy"

# import jax
# jax.config.update("jax_check_tracer_leaks", True)

# seed for reproducibility
set_seed(42)  # e.g. `set_seed(42)` for fixed seed

# load and wrap the Isaac Gym environment
env = make_env(task_name="Cartpole", headless=False, num_envs=1)
env = wrap_env(env, wrapper='isaacgym-preview4')
env = ImageObservationWrapper(env)
print(env.observation_space)

device = env.device

# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=1, num_envs=env.num_envs, device=device)

models = {}

sequence_length = 15
num_action_tokens = 11
layer_size = 256
vocab_size = 512
num_image_tokens = 81
rtx = RT1(
    num_image_tokens=num_image_tokens,
    num_action_tokens=num_action_tokens,
    layer_size=layer_size,
    vocab_size=vocab_size,
    # Use token learner to reduce tokens per image to 81.
    use_token_learner=True,
    # RT-1-X uses (-2.0, 2.0) instead of (-1.0, 1.0).
    world_vector_range=(-2.0, 2.0),
    
)

models["policy"] = RT1Policy(env.observation_space, env.action_space, device, rtx, batch_size=1, seqlen=sequence_length, task="balance the pole")
models["value"] = RT1Critic(env.observation_space, env.action_space, device, rtx, batch_size=1, seqlen=sequence_length, task="balance the pole") 

# instantiate models' state dict
for role, model in models.items():
    model.init_state_dict(role, ckpt="improve/custom/models/rt_1_x_jax/")

cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 1  # memory_size
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
cfg["state_preprocessor"] = RTXPreprocessor
cfg["state_preprocessor_kwargs"] = {
    "batch_size": 1, "seqlen": 15, "device": device}
# cfg["value_preprocessor"] = RunningStandardScaler
# cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

cfg["experiment"] = {
    "write_interval": 1,
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

# start training
trainer.train()


# start evaluation
# trainer.eval()
