import numpy as np
from improve.custom.models.rt1 import tokenize_action
import jax
import flax
import jax.numpy as jnp
import optax
from flax import struct
from typing import Any

from flax.training import checkpoints


@struct.dataclass
class TrainState:
    """Training state for PPO update."""
    step: int
    params: Any
    batch_stats: Any

    policy_opt_state: optax.OptState
    critic_opt_state: optax.OptState


def create_train_state(variables, policy_optimizer, critic_optimizer):
    params = variables["params"]
    batch_stats = variables["batch_stats"]

    train_state = TrainState(
        step=0,
        params=params,
        batch_stats=batch_stats,
        policy_opt_state=policy_optimizer.init(params),
        critic_opt_state=critic_optimizer.init(params),
    )
    return train_state


def train(batch, state, policy_model, critic_model, policy_optimizer, critic_optimizer, rng,
          ratio_clip, get_entropy, entropy_loss_scale, value_loss_scale, clip_predicted_values, value_clip):
    """Perform a single training step."""
    rng, loss_rng = jax.random.split(rng)

    def loss_fn(policy_params, critic_params):
        # Compute ppo update loss
        policy_vars = {"params": policy_params,
                       "batch_stats": state.policy_batch_stats}
        critic_vars = {"params": critic_params,
                       "batch_stats": state.critic_batch_stats}

        p_loss, (entropy_loss, kl_div, new_policy_bs) = policy_loss_fn(
            model=policy_model,
            batch=batch,
            variables=policy_vars,
            rng=loss_rng,
            ratio_clip=ratio_clip,
            get_entropy=get_entropy,
            entropy_loss_scale=entropy_loss_scale,
        )

        # --- critic (value) loss ---
        v_loss, new_critic_bs = critic_loss_fn(
            model=critic_model,
            batch=batch,
            variables=critic_vars,
            rng=loss_rng,
            value_loss_scale=value_loss_scale,
            clip_predicted_values=clip_predicted_values,
            value_clip=value_clip,
        )

        total_loss = p_loss + v_loss

        new_batch_stats = {
            "policy":  new_policy_bs,
            "critic":  new_critic_bs,
        }
        # new_rngs = {
        #     "policy": new_policy_rng,
        #     "critic": new_critic_rng,
        # }

        aux = {
            "p_loss":       p_loss,
            "v_loss":       v_loss,
            "entropy_loss": entropy_loss,
            "kl":           kl_div,
            "new_batch_stats": new_batch_stats,
            # "new_rngs":        new_rngs,
        }

        return total_loss, aux

    grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)
    (total_loss, aux), (p_grads, v_grads) = grad_fn(
        state.policy_params, state.critic_params
    )

    # 4) apply optimizer updates
    policy_updates, new_policy_opt_state = policy_optimizer.update(
        p_grads, state.policy_opt_state, state.policy_params
    )
    new_policy_params = optax.apply_updates(
        state.policy_params, policy_updates)

    critic_updates, new_critic_opt_state = critic_optimizer.update(
        v_grads, state.critic_opt_state, state.critic_params
    )
    new_critic_params = optax.apply_updates(
        state.critic_params, critic_updates)

    new_state = state.replace(
        step=state.step + 1,
        policy_params=new_policy_params,
        critic_params=new_critic_params,
        policy_opt_state=new_policy_opt_state,
        critic_opt_state=new_critic_opt_state,
        policy_batch_stats=aux["new_batch_stats"]["policy"],
        critic_batch_stats=aux["new_batch_stats"]["critic"],
    )

    metrics = {
        "policy_loss":  aux["p_loss"],
        "value_loss":   aux["v_loss"],
        "entropy_loss": aux["entropy_loss"],
        "kl_divergence": aux["kl"],
    }

    return new_state, metrics


def policy_loss_fn(model, batch, variables, rng, ratio_clip, get_entropy, entropy_loss_scale):
    states = batch['states']
    action = batch['actions']
    sampled_log_prob = batch['log_prob']
    sampled_advantages = batch['advantages']

    bs = states['image'].shape[0]
    seqlen = states['image'].shape[1]

    rng, params_rng = jax.random.split(rng)
    rng, dropout_rng = jax.random.split(rng)
    rng, sd_rng = jax.random.split(rng)
    rng, random_rng = jax.random.split(rng)

    logits, new_variables = model.apply(
        variables,
        obs=states,
        act=None,
        act_tokens=jnp.zeros((1, 6, 11)),
        train=True,
        mutable=["batch_stats"],
        rngs={
            "params": params_rng,
            "dropout": dropout_rng,
            "random": random_rng,
        },
    )

    vocab_size = model.vocab_size
    num_image_tokens = model.num_image_tokens
    num_action_tokens = model.num_action_tokens
    time_step_tokens = num_image_tokens + num_action_tokens
    logits = jnp.reshape(logits, (bs, seqlen, time_step_tokens, vocab_size))
    action_logits = logits[:, -1, ...]
    action_logits = action_logits[:, model.num_image_tokens - 1: -1]
    action_logits = jnp.reshape(action_logits, (bs, -1))
    logp = jax.nn.log_softmax(action_logits)

    ratio = logp - sampled_log_prob
    kl_divergence = ((jnp.exp(ratio) - 1) - ratio).mean()

    # compute policy loss
    ratio = jnp.exp(logp - sampled_log_prob)
    surrogate = sampled_advantages * ratio
    surrogate_clipped = sampled_advantages * \
        jnp.clip(ratio, 1.0 - ratio_clip, 1.0 + ratio_clip)

    # compute entropy loss
    entropy_loss = 0
    if entropy_loss_scale:
        entropy_loss = -entropy_loss_scale * \
            get_entropy(jnp.zeros((bs, 1), dtype=jnp.float32),
                        role="policy").mean()

    return -jnp.minimum(surrogate, surrogate_clipped).mean(), \
        (kl_divergence, entropy_loss, new_variables)


def critic_loss_fn(model, batch, variables, rng, value_loss_scale, clip_predicted_values, value_clip):
    states = batch['states']

    sampled_values = batch['values']
    sampled_returns = batch['returns']

    bs = states['image'].shape[0]
    seqlen = states['image'].shape[1]

    rng, params_rng = jax.random.split(rng)
    rng, dropout_rng = jax.random.split(rng)
    rng, random_rng = jax.random.split(rng)

    value, new_variables = model.apply(
        variables,
        obs=states,
        act=None,
        act_tokens=jnp.zeros((1, 6, 11)),
        train=True,
        mutable=["batch_stats"],
        rngs={
            "params": params_rng,
            "dropout": dropout_rng,
            "random": random_rng,
        },
    )

    value = jnp.reshape(value, (bs, -1))

    if clip_predicted_values:
        value = sampled_values + \
            jnp.clip(value - sampled_values, -
                     value_clip, value_clip)
    return value_loss_scale * ((sampled_returns - value) ** 2).mean(), new_variables
