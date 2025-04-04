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
from functools import partial


from improve.custom.models import rt1, base

def _categorical(net_output, unnormalized_log_prob, taken_actions, key):
    # normalize
    if unnormalized_log_prob:
        logits = net_output - jax.scipy.special.logsumexp(net_output, axis=-1, keepdims=True)
        # probs = jax.nn.softmax(logits)
    else:
        probs = net_output / net_output.sum(-1, keepdims=True)
        eps = jnp.finfo(probs.dtype).eps
        logits = jnp.log(probs.clip(min=eps, max=1 - eps))

    # sample actions
    actions = jax.random.categorical(key, logits, axis=-1, shape=None)

    # log of the probability density function
    taken_actions = actions if taken_actions is None else taken_actions.astype(jnp.int32).reshape(-1)
    log_prob = jax.nn.log_softmax(logits)[jnp.arange(taken_actions.shape[0]), taken_actions]

    return actions.reshape(-1, 1), log_prob.reshape(-1, 1) # [-2, 2] for RT2X?
    
@jax.jit
def _entropy(logits):
    logits = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    logits = logits.clip(min=jnp.finfo(logits.dtype).min)
    p_log_p = logits * jax.nn.softmax(logits)
    return -p_log_p.sum(-1)

class RT1Policy(base.Model):
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

        # Update the state_dict with the new batch_stats
        if self.training:
            self.state_dict = self.state_dict.replace(
                params=new_state
            )

        return output_logits

    def init_state_dict(self, role, inputs={}, key=None, ckpt=""):
        if not ckpt:
            super().init_state_dict(role, inputs, key)
        else:
            state_dict = checkpoints.restore_checkpoint(ckpt, None)
            state_dict['params']['image_tokenizer'] = freeze(state_dict['params']['image_tokenizer'])
            variables = {
                'params': state_dict['params'],
                'batch_stats': state_dict['batch_stats'],
            }

        self.state_dict = base.StateDict.create(
            apply_fn=self.apply, params=variables
        )
        
    def act(self, inputs, role='', params=None):
        observation = {"image": inputs["states"], "natural_language_embedding": self.embeds}
        self.rng, rng = jax.random.split(self.rng)
        
        output_logits = self(observation, rng=rng, params=params)
        time_step_tokens = self.model.num_image_tokens + self.model.num_action_tokens
        output_logits = jnp.reshape(
            output_logits, (self.batch_size, self.seqlen, time_step_tokens, -1)
        )
        action_logits = output_logits[:, -1, ...]
        action_logits = action_logits[:, self.model.num_image_tokens - 1 : -1]

        ### TODO: specifically for Cartpole
        action_logits = action_logits[:, 0]  # shape (bs, num_actions)
        actions, log_prob = _categorical(
            net_output=action_logits, unnormalized_log_prob=True, taken_actions=None, key=self.rng
        )
        actions = jnp.round(actions)

        outputs = {}
        outputs['net_output'] = action_logits
        outputs["stddev"] = jnp.zeros_like(log_prob)

        # # only select the 7 DOF action logits
        # action_logits = action_logits[:, :7]

        # actions, log_prob = _categorical(
        #     net_output=action_logits, unnormalized_log_prob=True, taken_actions=None, key=self.rng
        # )
        # print("Actions shape:", actions.shape)

        # outputs = {}
        # outputs['net_output'] = action_logits
        # outputs["stddev"] = jnp.full_like(log_prob, jnp.nan)

        return actions, log_prob, outputs

    def get_entropy(self, stddev: jax.Array, role: str = "") -> jax.Array: # replace with discrete entropy
        """Compute and return the entropy of the model

        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        :return: Entropy of the model
        :rtype: jax.Array
        """
        return _entropy(stddev)