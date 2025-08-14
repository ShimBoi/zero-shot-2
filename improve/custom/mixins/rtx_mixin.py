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

class RTXMixin:

    def __init__(self, task="", clip_actions=False, clip_values=False, reduction="sum"):
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

        self.rng = config.jax.key
        self._jitted_apply = jax.jit(self.forward, static_argnames=("role"))

        self.hist = deque(maxlen=self.seqlen)
        self.num_image_history = 0

        self.llm = hub.load(
            "https://tfhub.dev/google/universal-sentence-encoder-large/5"
        )

        ### IMPLEMENT CACHING
        embeds = self.llm([task]).numpy()
        self.embeds = jnp.array(embeds)
        self.embeds = jnp.expand_dims(self.embeds, 1)
        self.embeds = jnp.repeat(self.embeds, self.seqlen, axis=1) # leave batch for later

        flax.linen.Module.__post_init__(self)

    def _add_to_history(self, image) -> None:
        self.hist.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.seqlen)

    def _obtain_history(self):
        observation = jnp.stack(self.hist, axis=1)  # [: -self.seqlen]
        if observation.shape[1] < self.seqlen:
            pad = jnp.zeros(
                (
                    self.batch_size,
                    self.seqlen - observation.shape[1],
                    *observation.shape[2:],
                )
            )
            observation = jnp.concatenate([pad, observation], axis=1)
        return observation[:, -self.seqlen :]

    def forward(self, params, inputs, role, rng):
        rng, params_rng = jax.random.split(rng)
        rng, dropout_rng = jax.random.split(rng)
        rng, sd_rng = jax.random.split(rng)
        rng, random_rng = jax.random.split(rng)

        (logits, values, outputs), new_variables = self.apply(
            params,
            obs=inputs,
            act=None,
            act_tokens=self.act_tokens, # since jitted, this will stay constant even if changed outside
            train=True,
            mutable=["batch_stats"],
            rngs={
                "params": params_rng,
                "dropout": dropout_rng,
                "random": random_rng,
            },
        )

        bs = inputs["image"].shape[0]
        seqlen = inputs["image"].shape[1]    

        num_image_tokens = self.num_image_tokens
        num_action_tokens = self.num_action_tokens
        time_step_tokens = num_image_tokens + num_action_tokens
        logits = jnp.reshape(logits, (bs, seqlen, time_step_tokens, self.vocab_size))
        logits = logits[:, -1, num_image_tokens - 1 : -1] # [bs, num_action_tokens, vocab_size]

        # only care about action tokens [1:8]
        logits = logits[:, 1:8] # [bs, 7, vocab_size]

        # Sample actions from the logits    
        rng, log_probs_rng = jax.random.split(rng)
        log_probs_all = jax.nn.log_softmax(logits, axis=-1)
        action_token = jax.random.categorical(log_probs_rng, logits, axis=-1)
        action_token_expanded = jnp.expand_dims(action_token, axis=-1)  # [bs, 7, 1]
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

    def act(self, inputs, role="", params=None, update=False):
        ### TODO: implement not adding to history if updating
        if not update:
            pass

        image = inputs["states"].reshape((-1, *self.observation_space.shape))
        image = copy.deepcopy(inputs["states"])
        image = tf.image.resize(image, (300, 300)).numpy() / 225.0
        self.batch_size = image.shape[0]

        self._add_to_history(image)
        images = self._obtain_history()

        embeds = jnp.repeat(self.embeds, self.batch_size, axis=0)
        observation = {"image": images, "natural_language_embedding": embeds}

        params = self.state_dict.params if params is None else params
        params["batch_stats"] = self.batch_stats

        actions, log_prob, outputs = self._jitted_apply(
            params,
            observation,
            role,
            self.rng
        )

        if self._d_clip_values:
            values = jnp.clip(outputs["values"], a_min=self._d_clip_values_min, a_max=self._d_clip_values_max)
            outputs["values"] = values

        # update rngs and batch stats
        self.rng = jax.random.fold_in(self.rng, self.num_image_history)
        self.batch_stats = outputs["batch_stats"]

        return actions, log_prob, outputs

