import copy
from typing import Union, Tuple, Optional
import flax
from flax import linen as nn
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from flax.core import freeze, unfreeze

from improve.custom.models import rt1, base


class RT1Critic(base.Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        rt1_model = None,
        batch_size=1,
        seqlen=15,
        task="",
        **kwargs,
    ):
        base.Model.__init__(self, observation_space, action_space, device, **kwargs)
        assert rt1_model is not None, "RT1 model must be provided"
        assert isinstance(rt1_model, nn.Module), "RT1 model must be a Flax nn.Module"

        self.model = rt1_model

        self.rng = jax.random.PRNGKey(0)
        self.seqlen = seqlen
        self.batch_size = batch_size

        self.llm = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        self.embeds = self.llm([task]).numpy()
        self.embeds = np.expand_dims(self.embeds, 1)
        self.embeds = np.repeat(self.embeds, self.seqlen, axis=1)
        self.embeds = np.repeat(self.embeds, self.batch_size, axis=0)
        self.embeds = jnp.array(self.embeds, dtype=jnp.float32)

        self._is_initialized = False

    def setup(self):
        self.value_head = nn.Dense(features=1)

    def __call__(self, obs, act=None, obs_tokens=None, act_tokens=None, rng=jax.random.PRNGKey(0), params=None):
        if not act_tokens:
            act_tokens = jnp.zeros((1, 6, 11))

        rng, params_rng = jax.random.split(rng)
        rng, dropout_rng = jax.random.split(rng)
        rng, random_rng = jax.random.split(rng)

        # assume obs has batch dimension
        def apply_model(params, obs, act_tokens, rngs):
            return self.model.apply(
                params,
                obs,
                act=act,
                act_tokens=act_tokens,
                train=self.training,
                rngs=rngs,
                mutable=['params', 'batch_stats'],
            )
        
        output_logits, new_state = jax.checkpoint(apply_model)(
            self.state_dict.params if not params else params,
            obs,
            act_tokens=act_tokens,
            rngs={
                "params": params_rng,
                "dropout": dropout_rng,
                "random": random_rng
            },
        )

        # value = self.value_head(output_logits)
        value = jnp.mean(output_logits, axis=(1, 2), keepdims=True)
        value = jnp.squeeze(value, axis=-1)

        # Update the state_dict with the new batch_stats
        if self.training:
            self.state_dict = self.state_dict.replace(
                params=new_state
            )

        return value

    def init_state_dict(self, role, inputs={}, key=None, ckpt=""):
        if not ckpt:
            super().init_state_dict(role, inputs, key)
        else:
            state_dict = checkpoints.restore_checkpoint(ckpt, None)
            state_dict['params']['image_tokenizer'] = freeze(state_dict['params']['image_tokenizer'])
            # add dense layer to state dict
            variables = {
                'params': state_dict['params'],
                'batch_stats': state_dict['batch_stats'],
            }
        self.state_dict = base.StateDict.create(
            apply_fn=self.apply, params=variables
        )
        
    def act(self, inputs, role='', params=None):
        observation = {"image": inputs['states'], "natural_language_embedding": self.embeds}
        self.rng, rng = jax.random.split(self.rng)
        
        value = self(observation, rng=rng, params=params)
        
        return value, None, {}