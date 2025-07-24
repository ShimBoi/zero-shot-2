from improve.custom.memory.custom import CustomMemory
from improve.custom.trainers.sequential import SequentialTrainer
from improve.custom.preprocessor.rtx_preprocessor import RTXPreprocessor
from improve.utils.rtxFactory import createActorCriticModel
from improve.custom.env.env_wrapper import wrap_env
from improve.custom.env.loader import make_env
from improve.custom.algo.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.utils import set_seed
from skrl import config
import jax.numpy as jnp
import flax.linen as nn
import tensorflow_hub as hub
import numpy as np
import isaacgym
import jax
import functools
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

if not hasattr(functools, "cache"):
    functools.cache = functools.lru_cache

jax.config.update("jax_enable_x64", False)


config.jax.backend = "jax"  # or "numpy"
set_seed(42)

cfg = PPO_DEFAULT_CONFIG.copy()
cfg["sequence_length"] = 15
cfg["num_envs"] = 20
cfg["mini_batches"] = 4
cfg["rollouts"] = 100

env = make_env(task_name="FrankaCubePickCamera",
               headless=True, num_envs=cfg["num_envs"])
env = wrap_env(env, wrapper='isaacgym-preview4')
print(env.observation_space, env.action_space)

cfg["observation_space"] = env.observation_space
cfg["action_space"] = env.action_space
cfg["device"] = env.device
cfg["state_preprocessor"] = RTXPreprocessor
cfg["state_preprocessor_kwargs"] = {
    "batch_size": cfg["num_envs"],
    "seqlen": cfg["sequence_length"],
    "device": cfg["device"]
}

cfg["eps"] = 1e-6
cfg["learning_rate"] = 3e-5
cfg["ckpt"] = "improve/custom/models/rt_1_x_jax/"

memory = \
    CustomMemory(
        memory_size=cfg["rollouts"],
        num_envs=cfg["num_envs"]
    )

model = createActorCriticModel(cfg["ckpt"])
task = "pick up the cube"
llm = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
embeds = llm([task]).numpy()
embeds = np.expand_dims(embeds, 1)
embeds = np.repeat(embeds, 15, axis=1)
embeds_rollouts = jnp.array(
    np.repeat(embeds, cfg["num_envs"], axis=0), dtype=jnp.float32)
embeds_update = jnp.array(
    np.repeat(embeds, cfg["mini_batches"], axis=0), dtype=jnp.float32)

del llm

cfg["embeds_rollouts"] = embeds_rollouts
cfg["embeds_update"] = embeds_update
print("Embeds shape:", embeds_rollouts.shape, embeds_update.shape)

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
