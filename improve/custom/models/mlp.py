import copy
from flax.training import checkpoints
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence, Callable


def gaussian_log_prob(mu, sigma, actions):
    pre_sum = -0.5 * (((actions - mu) / sigma) ** 2 + 2 *
                      jnp.log(sigma) + jnp.log(2 * jnp.pi))
    return pre_sum.sum(axis=-1)


class MLPActorCritic(nn.Module):
    obs_dim: int
    act_dim: int
    hidden_units: Sequence[int] = (256, 128, 64)
    activation: Callable = nn.elu
    fixed_log_std: float = -1.0

    @nn.compact
    def __call__(self, x):
        for units in self.hidden_units:
            x = nn.Dense(units)(x)
            x = self.activation(x)

        mu = nn.Dense(self.act_dim)(x)
        v = nn.Dense(1)(x)

        log_std = self.param('log_std',
                             lambda key: jnp.full((self.act_dim,), self.fixed_log_std))
        sigma = jnp.exp(log_std)
        return mu, sigma, v.squeeze(-1)


class MLPActorCriticPolicy:
    def __init__(
        self,
        checkpoint_path=None,
        model=None,
        variables=None,
        rng=None,
        obs_dim=20,
        act_dim=4,
    ):
        self.model = model or MLPActorCritic(obs_dim=obs_dim, act_dim=act_dim)
        self._checkpoint_path = checkpoint_path
        self._value = None

        self._run_action_inference_jit = jax.jit(self._run_action_inference)

        if rng is not None:
            self.rng = rng
        else:
            self.rng = jax.random.PRNGKey(0)

        if variables:
            self.variables = variables
        else:
            dummy_obs = jnp.ones((1, obs_dim))
            init_rngs = {"params": jax.random.PRNGKey(0)}
            self.variables = self.model.init(init_rngs, dummy_obs)
            if checkpoint_path:
                ckpt = checkpoints.restore_checkpoint(
                    checkpoint_path, target=None)
                if "params" in ckpt:
                    self.variables = {"params": ckpt["params"]}

    def _run_action_inference(self, obs, rng):
        mu, sigma, value = self.model.apply(self.variables, obs)

        actions = mu + sigma * jax.random.normal(rng, mu.shape)

        logp = gaussian_log_prob(mu, sigma, actions)
        return actions, logp, value

    def action(self, obs):
        if obs.ndim == 1:
            obs = jnp.expand_dims(obs, 0)  # make batch dimension

        self.rng, rng = jax.random.split(self.rng)
        actions, logp, self._value = self._run_action_inference_jit(obs, rng)
        return actions, logp

    def value(self):
        return self._value

    def get_entropy(self, logits: jax.Array, role: str = "") -> jax.Array:
        return _entropy(logits)


@jax.jit
def _entropy(logits):
    logits = logits - \
        jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    logits = logits.clip(min=jnp.finfo(logits.dtype).min)
    p_log_p = logits * jax.nn.softmax(logits)
    return -p_log_p.sum(-1)
