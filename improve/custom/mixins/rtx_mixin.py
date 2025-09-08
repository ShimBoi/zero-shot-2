from collections import deque
import copy
import flax
import gymnasium
from improve.custom.models.rt1.utils import detokenize_action
import jax
import jax.numpy as jnp
from skrl import config
import tensorflow as tf
import tensorflow_hub as hub
import json
import os

class RTXMixin:

    def __init__(self, clip_actions=False, clip_values=False, reduction="sum"):
        self._a2c_clip_actions = clip_actions and isinstance(self.action_space, gymnasium.Space)

        if self._a2c_clip_actions:
            self._a2c_clip_actions_min = jnp.array(self.action_space.low, dtype=jnp.float32)
            self._a2c_clip_actions_max = jnp.array(self.action_space.high, dtype=jnp.float32)
        else:
            self._a2c_clip_actions_min = -jnp.inf
            self._a2c_clip_actions_max = jnp.inf

        # Reduction setup
        if reduction not in ["mean", "sum", "prod", "none"]:
            raise ValueError("reduction must be one of 'mean', 'sum', 'prod' or 'none'")
        self._a2c_reduction = (
            jnp.mean if reduction == "mean"
            else jnp.sum if reduction == "sum" 
            else jnp.prod if reduction == "prod" 
            else None
        )

        ### VALUE ARGS
        self._d_clip_values = clip_values and isinstance(self.action_space, gymnasium.Space)

        # NOTE: why clip by action space??
        if self._d_clip_values:
            self._d_clip_values_min = jnp.array(self.action_space.low, dtype=jnp.float32)
            self._d_clip_values_max = jnp.array(self.action_space.high, dtype=jnp.float32)

        flax.linen.Module.__post_init__(self)

    def forward(self, params, inputs, role, rng):
        rng, params_rng = jax.random.split(rng)
        rng, dropout_rng = jax.random.split(rng)
        rng, sd_rng = jax.random.split(rng)
        rng, random_rng = jax.random.split(rng)

        bs = inputs["image"].shape[0]
        seqlen = inputs["image"].shape[1]    

        (logits, values, outputs), new_variables = self.apply(
            params,
            obs=inputs,
            act=None,
            act_tokens=jnp.zeros((1, seqlen, self.num_action_tokens)),
            train=True,
            mutable=["batch_stats"],
            rngs={
                "params": params_rng,
                "dropout": dropout_rng,
                "random": random_rng,
            },
        )


        num_image_tokens = self.num_image_tokens
        num_action_tokens = self.num_action_tokens
        time_step_tokens = num_image_tokens + num_action_tokens
        logits = jnp.reshape(logits, (bs, seqlen, time_step_tokens, self.vocab_size))
        logits = logits[:, -1, num_image_tokens - 1 : -1] # [bs, num_action_tokens, vocab_size]

        # Sample actions from the logits    
        rng, log_probs_rng = jax.random.split(rng)
        action_token = jax.random.categorical(log_probs_rng, logits, axis=-1)

        # only care about action tokens [1:8]
        clipped_logits = logits[:, 1:8] # [bs, 7, vocab_size]
        clipped_action_tokens = action_token[:, 1:8]  # [bs, 7]

        log_probs_all = jax.nn.log_softmax(clipped_logits, axis=-1)
        action_token_expanded = jnp.expand_dims(clipped_action_tokens, axis=-1)  # [bs, 7, 1]
        log_probs = jnp.take_along_axis(log_probs_all, action_token_expanded, axis=-1).squeeze(axis=-1)  # [bs, 7]

        actions_dict = detokenize_action(
            action_token, self.vocab_size, self.world_vector_range)
        
        # only select the world vector, rotation delta, gripper
        action = jnp.concatenate(
            [
                actions_dict["world_vector"],
                actions_dict["rotation_delta"],
                actions_dict["gripper_closedness_action"],
            ],
            axis=-1,
        )

        action = jnp.clip(
            action,
            a_min=self._a2c_clip_actions_min, 
            a_max=self._a2c_clip_actions_max
        )

        # get value of last state
        values = values[:, -1, :]  # [bs, 1]

        outputs = {}
        outputs["values"] = values
        outputs["batch_stats"] = new_variables["batch_stats"]

        return action, log_probs, outputs
    
    def act(self, inputs, role="", params=None, rng=None):
        actions, log_prob, outputs = self.forward(
            params,
            inputs,
            role,
            rng
        )

        actions = jnp.clip(
            actions,
            a_min=self._a2c_clip_actions_min, 
            a_max=self._a2c_clip_actions_max
        )

        if self._a2c_reduction:
            log_prob = self._a2c_reduction(log_prob, axis=-1)
        if log_prob.ndim != actions.ndim:
            log_prob = jnp.expand_dims(log_prob, -1)

        if self._d_clip_values:
            values = jnp.clip(outputs["values"], a_min=self._d_clip_values_min, a_max=self._d_clip_values_max)
            outputs["values"] = values

        # compatibility
        outputs["stddev"] = jnp.zeros_like(outputs["values"])

        return actions, log_prob, outputs