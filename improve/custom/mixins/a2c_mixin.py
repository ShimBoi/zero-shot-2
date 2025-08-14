from typing import Any, Mapping, Optional, Tuple, Union
from functools import partial
import gymnasium

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

# Assuming you have the config from SKRL
# from skrl import config


# JIT-compiled helper functions (similar to SKRL's approach)
@partial(jax.jit, static_argnames=("reduction",))
def _gaussian_policy(
    loc, log_std, log_std_min, log_std_max, clip_actions_min, clip_actions_max, 
    taken_actions, key, reduction
):
    # clamp log standard deviations
    log_std = jnp.clip(log_std, a_min=log_std_min, a_max=log_std_max)
    scale = jnp.exp(log_std)
    
    # sample actions
    actions = jax.random.normal(key, loc.shape) * scale + loc
    
    # clip actions
    actions = jnp.clip(actions, a_min=clip_actions_min, a_max=clip_actions_max)
    
    # log probability
    taken_actions = actions if taken_actions is None else taken_actions
    log_prob = -jnp.square(taken_actions - loc) / (2 * jnp.square(scale)) - jnp.log(scale) - 0.5 * jnp.log(2 * jnp.pi)
    
    if reduction is not None:
        log_prob = reduction(log_prob, axis=-1)
    if log_prob.ndim != actions.ndim:
        log_prob = jnp.expand_dims(log_prob, -1)
        
    return actions, log_prob, log_std, scale


@jax.jit
def _entropy(scale):
    return 0.5 + 0.5 * jnp.log(2 * jnp.pi) + jnp.log(scale)


class A2CMixin:
    """A2C mixin that combines both policy and value functionality"""
    
    def __init__(
        self,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20,
        max_log_std: float = 2,
        reduction: str = "sum",
        clip_values: bool = True,
        role: str = "",
    ) -> None:
        """A2C mixin model (combines stochastic policy + deterministic value)

        :param clip_actions: Flag to indicate whether actions should be clipped (default: ``False``)
        :type clip_actions: bool, optional
        :param clip_log_std: Flag to indicate whether log standard deviations should be clipped (default: ``True``)
        :type clip_log_std: bool, optional
        :param min_log_std: Minimum value of log standard deviation (default: ``-20``)
        :type min_log_std: float, optional
        :param max_log_std: Maximum value of log standard deviation (default: ``2``)
        :type max_log_std: float, optional
        :param reduction: Reduction method for log probability (default: ``"sum"``)
        :type reduction: str, optional
        :param role: Role played by the model (default: ``""``)
        :type role: str, optional
        """

        ### POLICY ARGS
        # Action clipping setup
        self._a2c_clip_actions = clip_actions and isinstance(self.action_space, gymnasium.Space)
        
        if self._a2c_clip_actions:
            self._a2c_clip_actions_min = jnp.array(self.action_space.low, dtype=jnp.float32)
            self._a2c_clip_actions_max = jnp.array(self.action_space.high, dtype=jnp.float32)
        else:
            self._a2c_clip_actions_min = -jnp.inf
            self._a2c_clip_actions_max = jnp.inf

        # Log std clipping setup
        self._a2c_clip_log_std = clip_log_std
        if self._a2c_clip_log_std:
            self._a2c_log_std_min = min_log_std
            self._a2c_log_std_max = max_log_std
        else:
            self._a2c_log_std_min = -jnp.inf
            self._a2c_log_std_max = jnp.inf

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

        # Random key management
        self._a2c_i = 0
        # You'll need to set this from your config or pass it in
        self._a2c_key = jax.random.PRNGKey(0)  # Replace with config.jax.key

        ### CUSTOM ARGS
        self._role = role
        self._jitted_apply = jax.jit(self.forward, static_argnames=("role"))

        # Important for Flax
        flax.linen.Module.__post_init__(self)

    def forward(self, params, inputs, role):
        """Forward pass for the A2C mixin model"""
        with jax.default_device(self.device):
            self._a2c_i += 1
            subkey = jax.random.fold_in(self._a2c_key, self._a2c_i)
            inputs["key"] = subkey

        # Get mean actions, log std, and values from the model
        mean_actions, log_std, values, outputs = self.apply(params, inputs, role)

        # Sample actions and compute log probabilities
        actions, log_prob, log_std, stddev = _gaussian_policy(
            mean_actions,
            log_std,
            self._a2c_log_std_min,
            self._a2c_log_std_max,
            self._a2c_clip_actions_min,
            self._a2c_clip_actions_max,
            inputs.get("taken_actions", None),
            subkey,
            self._a2c_reduction,
        )

        # Store outputs
        outputs["mean_actions"] = mean_actions
        outputs["log_std"] = log_std
        outputs["stddev"] = stddev
        outputs["values"] = values

        return actions, log_prob, outputs

    def act(
        self,
        inputs: Mapping[str, Union[Union[np.ndarray, jax.Array], Any]],
        role: str = "",
        params: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, Union[jax.Array, None], Mapping[str, Union[jax.Array, Any]]]:
        """Act stochastically and compute value in response to environment state

        :param inputs: Model inputs with keys like "states" and optionally "taken_actions"
        :type inputs: dict
        :param role: Role played by the model (default: ``""``)
        :type role: str, optional
        :param params: Parameters for computation (default: ``None``)
        :type params: jnp.array, optional

        :return: Actions, log probabilities, and outputs dict containing values and other info
        :rtype: tuple of jax.Array, jax.Array, and dict
        """
        params = self.state_dict.params if params is None else params
        actions, log_prob, outputs = self._jitted_apply(params, inputs, role)

        if self._d_clip_values:
            values = jnp.clip(outputs["values"], a_min=self._d_clip_values_min, a_max=self._d_clip_values_max)
            outputs["values"] = values

        return actions, log_prob, outputs

    def get_entropy(self, stddev: jax.Array, role: str = "") -> jax.Array:
        """Compute and return the entropy of the policy

        :param stddev: Standard deviation of the policy
        :type stddev: jax.Array
        :param role: Role played by the model (default: ``""``)
        :type role: str, optional

        :return: Entropy of the policy
        :rtype: jax.Array
        """
        return _entropy(stddev)