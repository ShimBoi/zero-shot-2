
import copy

from flax.training import checkpoints
import jax
import jax.numpy as jnp
import improve.custom.models.rt1 as rt1
import improve.utils.helper as utils
from flax import linen as nn
from typing import Sequence


class RT1ActorCritic:
    """Runs inference with a RT-1 policy."""

    def __init__(
        self,
        checkpoint_path=None,
        model=rt1.RT1(),
        variables=None,
        seqlen=15,
        rng=None,
    ):
        """Initializes the policy.

        Args:
        checkpoint_path: A checkpoint point from which to load variables. Either
            this or variables must be provided.
        model: A nn.Module to use for the policy. Must match with the variables
            provided by checkpoint_path or variables.
        variables: If provided, will use variables instead of loading from
            checkpoint_path.
        seqlen: The history length to use for observations.
        rng: a jax.random.PRNGKey to use for the random number generator.
        """
        if not variables and not checkpoint_path:
            raise ValueError(
                'At least one of `variables` or `checkpoint_path` must be defined.'
            )
        self.model = model
        self._checkpoint_path = checkpoint_path
        self.seqlen = seqlen
        self._value = None

        self._run_action_inference_jit = jax.jit(self._run_action_inference)

        if rng is None:
            self.rng = jax.random.PRNGKey(0)
        else:
            self.rng = rng

        if variables:
            self.variables = variables
        else:
            state_dict = checkpoints.restore_checkpoint(checkpoint_path, None)
            obs = {
                "image": jnp.ones((8, 15, 300, 300, 3)),
                "natural_language_embedding": jnp.ones((8, 15, 512)),
            }

            rngs = {"params": jax.random.PRNGKey(
                0), "random": jax.random.PRNGKey(0)}
            act_tokens = jnp.zeros((1, 6, 11))
            init_vars = self.model.init(
                rngs=rngs, obs=obs, act=None, act_tokens=act_tokens, train=False)

            ckpt = checkpoints.restore_checkpoint(checkpoint_path, target=None)
            old_params = ckpt["params"]

            new_params = init_vars["params"]
            for module_name, module_params in old_params.items():
                new_params[module_name] = module_params

            stopped_params = utils.stop_gradient_subtree(
                new_params, freeze_keys=["image_tokenizer"]
            )
            variables = {
                'params': stopped_params,
                'batch_stats': state_dict['batch_stats'],
            }
            self.variables = variables

    def _run_action_inference(self, obs, rng):
        """A jittable function for running inference."""

        # We add zero action tokens so that the shape is (seqlen, 11).
        # Note that in the vanilla RT-1 setup, where
        # `include_prev_timesteps_actions=False`, the network will not use the
        # input tokens and instead uses zero action tokens, thereby not using the
        # action history. We still pass it in for simplicity.
        bs = obs['image'].shape[0]
        act_tokens = jnp.zeros((1, 6, 11))
        _, random_rng = jax.random.split(rng)

        output_logits, value = self.model.apply(
            self.variables,
            obs,
            act=None,
            act_tokens=act_tokens,
            train=False,
            rngs={'random': random_rng},
        )

        time_step_tokens = (
            self.model.num_image_tokens + self.model.num_action_tokens
        )
        output_logits = jnp.reshape(
            output_logits, (bs, self.seqlen, time_step_tokens, -1)
        )
        action_logits = output_logits[:, -1, ...]
        action_logits = action_logits[:, self.model.num_image_tokens - 1: -1]

        action_logp = jax.nn.softmax(action_logits)
        # sample from probability distribution
        logp = action_logp.reshape((bs, -1))
        # store as one hot or index on what is chosen
        action_token = jnp.argmax(action_logp, axis=-1)

        # Detokenize the full action sequence.
        detokenized = rt1.detokenize_action(
            action_token, self.model.vocab_size, self.model.world_vector_range
        )

        actions = jnp.concatenate([detokenized["world_vector"],
                                   detokenized["rotation_delta"],
                                   detokenized["gripper_closedness_action"]], axis=1)
        return actions, logp, value

    def action(self, observation):
        """Outputs the action given observation from the env."""
        # Assume obs has no batch dimensions.
        observation = copy.deepcopy(observation)

        # Jax does not support string types, so remove it from the dict if it
        # exists.
        if 'natural_language_instruction' in observation:
            del observation['natural_language_instruction']

        self.rng, rng = jax.random.split(self.rng)
        action, logp, self._value = self._run_action_inference_jit(
            observation, rng
        )
        return action, logp

    def value(self):
        return self._value

    # replace with discrete entropy
    def get_entropy(self, stddev: jax.Array, role: str = "") -> jax.Array:
        """Compute and return the entropy of the model

        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        :return: Entropy of the model
        :rtype: jax.Array
        """
        return _entropy(stddev)


@jax.jit
def _entropy(logits):
    logits = logits - \
        jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    logits = logits.clip(min=jnp.finfo(logits.dtype).min)
    p_log_p = logits * jax.nn.softmax(logits)
    return -p_log_p.sum(-1)
