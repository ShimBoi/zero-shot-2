
from typing import Any, Mapping, Optional, Tuple, Union
import gymnasium

import time
import jax
import jax.numpy as jnp
import optax
import numpy as np
from flax import linen as nn

import functools as ft
from skrl import config, logger
from improve.custom.algo.base import Agent
from skrl.memories.jax import Memory
from improve.custom.algo.mlp.ppo_update import create_train_state, train
from optax import multi_transform
from flax.traverse_util import flatten_dict, unflatten_dict
import tensorflow_datasets as tfds


# from skrl.resources.optimizers.jax import Adam

# fmt: off
# [start-config-dict-jax]
PPO_DEFAULT_CONFIG = {
    "rollouts": 16,                 # number of rollouts before updating
    "learning_epochs": 8,           # number of learning epochs during each update
    "mini_batches": 2,              # number of mini batches during each learning epoch

    "discount_factor": 0.99,        # discount factor (gamma)
    "lambda": 0.95,                 # TD(lambda) coefficient (lam) for computing returns and advantages

    "learning_rate": 1e-3,                  # learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler function (see optax.schedules)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})
    "value_preprocessor": None,             # value preprocessor class (see skrl.resources.preprocessors)
    "value_preprocessor_kwargs": {},        # value preprocessor's kwargs (e.g. {"size": 1})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0.5,              # clipping coefficient for the norm of the gradients
    "ratio_clip": 0.2,                  # clipping coefficient for computing the clipped surrogate objective
    "value_clip": 0.2,                  # clipping coefficient for computing the value loss (if clip_predicted_values is True)
    "clip_predicted_values": False,     # clip predicted values during value loss computation

    "entropy_loss_scale": 0.0,      # entropy loss scaling factor
    "value_loss_scale": 1.0,        # value loss scaling factor

    "kl_threshold": 0,              # KL divergence threshold for early stopping

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward
    "time_limit_bootstrap": False,  # bootstrap at timeout termination (episode truncation)

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": "auto",   # TensorBoard writing interval (timesteps)

        "checkpoint_interval": "auto",      # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}
# [end-config-dict-jax]
# fmt: on


# https://jax.readthedocs.io/en/latest/faq.html#strategy-1-jit-compiled-helper-function
@jax.jit
def _compute_gae(
    rewards: jax.Array,
    dones: jax.Array,
    values: jax.Array,
    next_values: jax.Array,
    discount_factor: float = 0.99,
    lambda_coefficient: float = 0.95,
) -> jax.Array:
    advantage = 0
    advantages = jnp.zeros_like(rewards)
    not_dones = jnp.logical_not(dones)
    memory_size = rewards.shape[0]

    # advantages computation
    for i in reversed(range(memory_size)):
        next_values = values[i + 1] if i < memory_size - 1 else next_values
        advantage = (
            rewards[i] - values[i] + discount_factor * not_dones[i] *
            (next_values + lambda_coefficient * advantage)
        )
        advantages = advantages.at[i].set(advantage)
    # returns computation
    returns = advantages + values
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return returns, advantages


class PPO(Agent):
    def __init__(
        self,
        model,
        memory: Optional[Union[Memory, Tuple[Memory]]] = None,
        observation_space: Optional[Union[int,
                                          Tuple[int], gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        device: Optional[Union[str, jax.Device]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        """Proximal Policy Optimization (PPO)

        https://arxiv.org/abs/1707.06347

        :param models: Models used by the agent
        :type models: dictionary of skrl.models.jax.Model
        :param memory: Memory to storage the transitions.
                       If it is a tuple, the first element will be used for training and
                       for the rest only the environment transitions will be added
        :type memory: skrl.memory.jax.Memory, list of skrl.memory.jax.Memory or None
        :param observation_space: Observation/state space or shape (default: ``None``)
        :type observation_space: int, tuple or list of int, gymnasium.Space or None, optional
        :param action_space: Action space or shape (default: ``None``)
        :type action_space: int, tuple or list of int, gymnasium.Space or None, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or jax.Device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict

        :raises KeyError: If the models dictionary is missing a required key
        """
        # _cfg = copy.deepcopy(PPO_DEFAULT_CONFIG)  # TODO: TypeError: cannot pickle 'jax.Device' object
        _cfg = PPO_DEFAULT_CONFIG
        _cfg.update(cfg if cfg is not None else {})

        self.cfg = _cfg

        self.model = model
        super().__init__(
            models={"model": self.model},
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=_cfg,
        )

        # checkpoint models
        self.checkpoint_modules["policy"] = self.model
        self.checkpoint_modules["value"] = self.model

        # broadcast models' parameters in distributed runs
        # if config.jax.is_distributed:
        #     logger.info(f"Broadcasting models' parameters")
        #     if self.policy is not None:
        #         self.policy.broadcast_parameters()
        #     if self.value is not None:
        #         self.value.broadcast_parameters()

        # configuration
        self._learning_epochs = self.cfg["learning_epochs"]
        self._mini_batches = self.cfg["mini_batches"]
        self._rollouts = self.cfg["rollouts"]
        self._rollout = 0

        self._grad_norm_clip = self.cfg["grad_norm_clip"]
        self._ratio_clip = self.cfg["ratio_clip"]
        self._value_clip = self.cfg["value_clip"]
        self._clip_predicted_values = self.cfg["clip_predicted_values"]

        self._value_loss_scale = self.cfg["value_loss_scale"]
        self._entropy_loss_scale = self.cfg["entropy_loss_scale"]

        self._kl_threshold = self.cfg["kl_threshold"]

        self._learning_rate = self.cfg["learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._state_preprocessor = self.cfg["state_preprocessor"]
        self._value_preprocessor = self.cfg["value_preprocessor"]

        self._discount_factor = self.cfg["discount_factor"]
        self._lambda = self.cfg["lambda"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._rewards_shaper = self.cfg["rewards_shaper"]
        self._time_limit_bootstrap = self.cfg["time_limit_bootstrap"]

        # set up preprocessors
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(
                **self.cfg["state_preprocessor_kwargs"])
        else:
            self._state_preprocessor = self._empty_preprocessor

        if self._value_preprocessor:
            self._value_preprocessor = self._value_preprocessor(
                **self.cfg["value_preprocessor_kwargs"])
        else:
            self._value_preprocessor = self._empty_preprocessor

        # construct optimizers (actor masks out value head, critic only updates value head)

        flat = flatten_dict(self.model.variables["params"], sep="/")
        labels = {}
        for path in flat:
            if path.startswith("value_head"):
                labels[path] = "critic"
            else:
                labels[path] = "actor"

        label_tree = unflatten_dict(labels, sep="/")

        self.optimizer = multi_transform(
            {"actor": optax.adam(1e-4),
             "critic": optax.adam(1e-3)},
            param_labels=label_tree
        )

        agent_create_train_state = ft.partial(
            create_train_state,
            optimizer=self.optimizer,
            variables=self.model.variables
        )
        create_train_state_jit = jax.jit(agent_create_train_state)

        self.state = create_train_state_jit()

        agent_train = ft.partial(train, model=self.model.model,
                                 optimizer=self.optimizer,
                                 ratio_clip=self._ratio_clip,
                                 get_entropy=self.model.get_entropy,
                                 entropy_loss_scale=self._entropy_loss_scale,
                                 value_loss_scale=self._value_loss_scale,
                                 clip_predicted_values=self._clip_predicted_values,
                                 value_clip=self._value_clip)
        self.jitted_train_step = jax.jit(agent_train)

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent"""
        super().init(trainer_cfg=trainer_cfg)

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(
                name="states", size=np.prod(self.observation_space.shape), dtype=jnp.float32)
            self.memory.create_tensor(
                name="actions", size=np.prod(self.action_space.shape), dtype=jnp.float32)
            self.memory.create_tensor(
                name="rewards", size=1, dtype=jnp.float32)
            self.memory.create_tensor(
                name="terminated", size=1, dtype=jnp.int8)
            self.memory.create_tensor(name="truncated", size=1, dtype=jnp.int8)
            self.memory.create_tensor(
                name="log_prob", size=1, dtype=jnp.float32)
            self.memory.create_tensor(name="values", size=1, dtype=jnp.float32)
            self.memory.create_tensor(
                name="returns", size=1, dtype=jnp.float32)
            self.memory.create_tensor(
                name="advantages", size=1, dtype=jnp.float32)
            self.memory.create_tensor(
                name="is_first", size=1, dtype=jnp.int8)
            self.memory.create_tensor(
                name="is_last", size=1, dtype=jnp.int8)

            # tensors sampled during training
            self._tensors_names = ["states", "actions", "terminated", "truncated",
                                   "log_prob", "values", "returns", "advantages", "is_first", "is_last"]

        # create temporary variables needed for storage and computation
        self._current_log_prob = None
        self._current_next_states = None

    def act(self, states: Union[np.ndarray, jax.Array], timestep) -> Union[np.ndarray, jax.Array]:
        """Process the environment's states to make a decision (actions) using the main policy

        :param states: Environment's states
        :type states: np.ndarray or jax.Array
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: np.ndarray or jax.Array
        """
        # sample random actions
        # TODO, check for stochasticity
        # if timestep < self._random_timesteps:
        #     return self.policy.random_act({"states": self._state_preprocessor(states)}, role="policy")

        actions, logp = self.model.action(states)

        if not self._jax:  # numpy backend
            actions = jax.device_get(actions)
            logp = jax.device_get(logp)

        self._current_log_prob = logp

        return actions, logp

    def record_transition(
        self,
        states: Union[np.ndarray, jax.Array],
        actions: Union[np.ndarray, jax.Array],
        rewards: Union[np.ndarray, jax.Array],
        next_states: Union[np.ndarray, jax.Array],
        terminated: Union[np.ndarray, jax.Array],
        truncated: Union[np.ndarray, jax.Array],
        is_first: Union[np.ndarray, jax.Array],
        is_last: Union[np.ndarray, jax.Array],
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        """Record an environment transition in memory

        :param states: Observations/states of the environment used to make the decision
        :type states: np.ndarray or jax.Array
        :param actions: Actions taken by the agent
        :type actions: np.ndarray or jax.Array
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: np.ndarray or jax.Array
        :param next_states: Next observations/states of the environment
        :type next_states: np.ndarray or jax.Array
        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: np.ndarray or jax.Array
        :param truncated: Signals to indicate that episodes have been truncated
        :type truncated: np.ndarray or jax.Array
        :param infos: Additional information about the environment
        :type infos: Any type supported by the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        super().record_transition(
            states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps
        )

        if self.memory is not None:
            self._current_next_states = next_states

            # reward shaping
            if self._rewards_shaper is not None:
                rewards = self._rewards_shaper(rewards, timestep, timesteps)

            # compute values
            values = self.model.value()

            if not self._jax:  # numpy backend
                values = jax.device_get(values)
            values = self._value_preprocessor(values, inverse=True)

            # time-limit (truncation) bootstrapping
            if self._time_limit_bootstrap:
                rewards += self._discount_factor * values * truncated

            log_prob = self._current_log_prob

            self.memory.add_samples(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
                log_prob=log_prob,
                values=values,
                is_first=is_first,
                is_last=is_last,
            )

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called before the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        raise NotImplementedError()

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        self._rollout += 1

        if not self._rollout % self._rollouts and timestep >= self._learning_starts:
            self._update(timestep, timesteps)

        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)

    def _update(self, timestep: int, timesteps: int) -> None:
        rng = self.model.rng

        _ = self.model.action(self._current_next_states)
        last_values = self.model.value()

        values = self.memory.get_tensor_by_name("values")

        returns, advantages = _compute_gae(
            rewards=self.memory.get_tensor_by_name("rewards"),
            dones=(self.memory.get_tensor_by_name(
                "terminated") | self.memory.get_tensor_by_name("truncated")),
            values=values,
            next_values=last_values,
            discount_factor=self._discount_factor,
            lambda_coefficient=self._lambda,
        )

        self.memory.set_tensor_by_name("returns", returns)
        self.memory.set_tensor_by_name("advantages", advantages)

        ds = self.memory.sample_all(sequence_length=1,
                                    names=self._tensors_names,
                                    minibatches=self._mini_batches,
                                    is_image_obs=False)
        for batch in tfds.as_numpy(ds):
            self.state, metrics_update = self.jitted_train_step(
                state=self.state, batch=batch, rng=rng
            )
            print(metrics_update)
