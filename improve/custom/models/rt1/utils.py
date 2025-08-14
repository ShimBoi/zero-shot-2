from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import copy


def tokenize_action(
    actions: Dict[str, jnp.ndarray],
    vocab_size: int,
    world_vector_range: Tuple[float, float] = (-1.0, 1.0),
) -> jnp.ndarray:
    """Tokenizes the action for the RT-1 task.

    <name>: <shape> <bounds>
    terminate_episode: (3,) int32,
      mode 0: terminate episode
      mode 1: arm + gripper

      mode 2: base
    world_vector: (3,) [-1.0, 1.0] (RT-1) or [-2.0, 2.0] (RT-1-X)
    rotation_delta: (3,) [-np.pi, np.pi]
    gripper_closedness_action: (1,) [-1, 1]
    base_displacement_vertical_rotation: (1,) [-np.pi, np.pi]
    base_displacement_vector: (2,) [-1.0, 1.0]

    Args:
      actions: The raw action dictionary.
      vocab_size: The vocab size of the tokenized actions.
      world_vector_range: The bounds to use for the world_vector token.

    Returns:
      the tokenized action.
    """
    action_tokens = []

    # Handle the discrete one first.
    terminate_episode = actions['terminate_episode']
    terminate_episode = jnp.argmax(terminate_episode, axis=-1)
    terminate_episode = jnp.expand_dims(terminate_episode, -1)
    terminate_episode = terminate_episode.astype(jnp.int32)
    action_tokens.append(terminate_episode)

    for act_name, act_min, act_max in [
        ('world_vector', world_vector_range[0], world_vector_range[1]),
        ('rotation_delta', -jnp.pi / 2, jnp.pi / 2),
        ('gripper_closedness_action', -1.0, 1.0),
        ('base_displacement_vertical_rotation', -jnp.pi, jnp.pi),
        ('base_displacement_vector', -1.0, 1.0),
    ]:
        act = actions[act_name]
        act = jnp.clip(act, act_min, act_max)
        act = (act - act_min) / (act_max - act_min)
        act = act * (vocab_size - 1)
        act = act.astype(jnp.int32)
        action_tokens.append(act)

    tokenized = jnp.concatenate(action_tokens, axis=-1)
    return tokenized


def detokenize_action(
    tokenized_actions: jnp.ndarray,
    vocab_size: int,
    world_vector_range: Tuple[float, float] = (-1.0, 1.0),
) -> Dict[str, jnp.ndarray]:
    """De-tokenizes the action for the RT-1 task.

    See `tokenize_action` for information on the action structure.

    Args:
      tokenized_actions: The tokenized action vector.
      vocab_size: The vocab size of the tokenized actions.
      world_vector_range: The bounds to use for the world_vector token.

    Returns:
      the detokenized action dictionary.
    """
    terminate_episode = tokenized_actions[:, 0]
    terminate_episode = jax.nn.one_hot(terminate_episode, 3)

    raw_actions = dict(
        world_vector=tokenized_actions[:, 1:4].astype(jnp.float32),
        rotation_delta=tokenized_actions[:, 4:7].astype(jnp.float32),
        gripper_closedness_action=tokenized_actions[:, 7:8].astype(
            jnp.float32),
        base_displacement_vertical_rotation=tokenized_actions[:, 8:9].astype(
            jnp.float32
        ),
        base_displacement_vector=tokenized_actions[:, 9:11].astype(
            jnp.float32),
    )

    act_dict = {'terminate_episode': terminate_episode.astype(jnp.int32)}
    for act_name, act_min, act_max in [
        ('world_vector', world_vector_range[0], world_vector_range[1]),
        ('rotation_delta', -jnp.pi / 2, jnp.pi / 2),
        ('gripper_closedness_action', -1.0, 1.0),
        ('base_displacement_vertical_rotation', -jnp.pi, jnp.pi),
        ('base_displacement_vector', -1.0, 1.0),
    ]:
        act = raw_actions[act_name]
        act = act / (vocab_size - 1)
        act = act * (act_max - act_min)
        act = act + act_min
        act_dict[act_name] = act

    return act_dict