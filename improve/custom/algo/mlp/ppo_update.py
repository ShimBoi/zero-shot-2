import numpy as np
from improve.custom.models.mlp import gaussian_log_prob
import jax
import flax
import jax.numpy as jnp
import optax
from flax import struct
from typing import Any


@struct.dataclass
class TrainState:
    """Training state for PPO update."""
    step: int
    params: Any
    optimizer: optax.OptState


def create_train_state(variables, optimizer):
    params = variables["params"]

    train_state = TrainState(
        step=0,
        params=params,
        optimizer=optimizer.init(params),
    )
    return train_state


def train(batch, state, model, optimizer, rng,
          ratio_clip, get_entropy, entropy_loss_scale,
          value_loss_scale, clip_predicted_values, value_clip):
    """Perform a single training step."""
    rng, loss_rng = jax.random.split(rng)

    def loss_fn(params):
        """Compute the PPO loss."""
        variables = {"params": params}
        obs = batch['states']
        sampled_log_prob = batch['log_prob']
        sampled_advantages = batch['advantages']
        sampled_values = batch['values']
        sampled_returns = batch['returns']
        sampled_actions = batch['actions']

        bs = obs.shape[0]

        (mu, sigma, value_pred) = model.apply(variables, obs)
        logp = gaussian_log_prob(mu, sigma, sampled_actions)

        mu, sigma, value_pred = model.apply(variables, obs)
        jax.debug.print("mu: {}", mu)
        jax.debug.print("sigma: {}", sigma)
        jax.debug.print("value_pred: {}", value_pred)
        jax.debug.print("logp: {}", logp)

        def _policy_loss_fn(logp, sampled_log_prob, sampled_advantages):
            """Compute the policy loss."""

            ratio = logp - sampled_log_prob
            kl_divergence = ((jnp.exp(ratio) - 1) - ratio).mean()

            ratio = jnp.exp(logp - sampled_log_prob)
            surrogate = sampled_advantages * ratio
            surrogate_clipped = sampled_advantages * \
                jnp.clip(ratio, 1.0 - ratio_clip, 1.0 + ratio_clip)

            entropy_loss = 0
            if entropy_loss_scale:
                entropy_loss = -entropy_loss_scale * \
                    get_entropy(jnp.zeros((bs, 1), dtype=jnp.float32),
                                role="policy").mean()

            return -jnp.minimum(surrogate, surrogate_clipped).mean(), \
                (kl_divergence, entropy_loss)

        def _critic_loss_fn(value_pred, sampled_values, sampled_returns):
            """Compute the critic loss."""
            value_pred = jnp.reshape(value_pred, (bs, -1))

            if clip_predicted_values:
                value_pred = sampled_values + \
                    jnp.clip(value_pred - sampled_values, -
                             value_clip, value_clip)
            return value_loss_scale * ((sampled_returns - value_pred) ** 2).mean()

        p_loss, (kl_div, entropy_loss) = _policy_loss_fn(
            logp, sampled_log_prob, sampled_advantages)
        v_loss = _critic_loss_fn(value_pred, sampled_values, sampled_returns)

        total_loss = p_loss + v_loss + entropy_loss

        aux = {
            "p_loss": p_loss,
            "v_loss": v_loss,
            "entropy_loss": entropy_loss,
            "kl_divergence": kl_div,
        }

        return total_loss, aux

    (total_loss, aux), grads = jax.value_and_grad(
        loss_fn, has_aux=True)(state.params)

    # 4) apply optimizer updates
    updates, opt_state = optimizer.update(
        grads, state.optimizer, params=state.params)

    new_params = optax.apply_updates(state.params, updates)

    new_state = state.replace(
        step=state.step + 1,
        params=new_params,
        optimizer=opt_state,
    )

    metrics = {
        "policy_loss":  aux["p_loss"],
        "value_loss":   aux["v_loss"],
        "entropy_loss": aux["entropy_loss"],
        "kl_divergence": aux["kl_divergence"],
    }

    return new_state, metrics
