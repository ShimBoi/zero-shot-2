from collections import deque
import copy
import gymnasium
from improve.custom.models.rt1.rt1 import RT1
from improve.custom.models.rt1.utils import detokenize_action, tokenize_action
import jax
import jax.numpy as jnp
from skrl import config
import tensorflow_hub as hub
import json
import os
from typing import Mapping, Optional, Union
from skrl.models.jax.base import Model, StateDict
import numpy as np

class RTXPPO:
    """
    A wrapper for the RT1 model that integrates RTX functionality.
    This class extends the RT1 model to support RTX-specific features.
    """

    def __init__(self, observation_space, action_space, device, task="", clip_actions=True, clip_values=False, reduction="sum", seqlen=15):

        self.rng = config.jax.key
        self.model = RT1()
        self.seqlen = seqlen
        self.hist = deque(maxlen=self.seqlen)
        self.num_image_history = 0
        self.trainig = False
        self._load_embeddings(task)

        ### RTX Mixin initialization
        self.device = device
        self.observation_space = observation_space
        self.action_space = action_space
        self._setup_mixin(clip_actions, clip_values, reduction)

        self._jitted_rollout = jax.jit(self._rollout)
        self._jitted_update = jax.jit(self._update)

    def _setup_mixin(self, clip_actions, clip_values, reduction):
        """Setup RTX mixin parameters."""

        self._a2c_clip_actions = clip_actions and isinstance(self.action_space, gymnasium.Space)
        self._d_clip_values = clip_values and isinstance(self.action_space, gymnasium.Space)

        if self._a2c_clip_actions:
            self._a2c_clip_actions_min = jnp.array(self.action_space.low, dtype=jnp.float32)
            self._a2c_clip_actions_max = jnp.array(self.action_space.high, dtype=jnp.float32)
        else:
            self._a2c_clip_actions_min = -jnp.inf
            self._a2c_clip_actions_max = jnp.inf

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

    def get_states(self):
        """Get the current state of the model's history."""
        history = self._obtain_history()
        bs = history.shape[0]
        return history.reshape(bs, -1)
    
    def _last_value_obtain_history(self, image):
        hist = copy.deepcopy(self.hist)
        self._add_to_history(image)
        observation = self._obtain_history()
        self.hist = hist
        return observation

    def _process_inputs_by_role(self, role, inputs):
        """Process inputs based on role type."""
        
        if role == "update":
            images = inputs["states"]
            images = jax.image.resize(images, (images.shape[0], images.shape[1], 300, 300, 3), method='linear')
            return images
        elif role in ["rollout", "last_value"]:
            image = self._preprocess_image(inputs["states"])
            self.batch_size = image.shape[0]
            
            if role == "rollout":
                self._add_to_history(image)
            elif role == "last_value":
                return self._last_value_obtain_history(image)
        
        return self._obtain_history()

    def _preprocess_image(self, states):
        """Common image preprocessing logic."""
        bs = states.shape[0]
        image = states.reshape((-1, *self.observation_space.shape))
        image = jax.image.resize(image, (bs, 300, 300, image.shape[-1]), method='linear') # env already does scaling / 255.0
        return image

    def _forward(self, params, observation, rng, train=False):
        rng, params_rng = jax.random.split(rng)
        rng, dropout_rng = jax.random.split(rng)
        rng, sd_rng = jax.random.split(rng)
        rng, random_rng = jax.random.split(rng)

        mutable = ["batch_stats"] if train else []
        bs, seqlen, *_ = observation["image"].shape
        (logits, values, outputs), new_variables = self.model.apply(
            params,
            obs=observation,
            act=None,
            act_tokens=jnp.zeros((bs, 6, 11)),
            train=train,
            mutable=mutable,
            rngs={
                "params": params_rng,
                "dropout": dropout_rng,
                "random": random_rng,
            },
        )

        num_image_tokens = self.model.num_image_tokens
        num_action_tokens = self.model.num_action_tokens
        time_step_tokens = num_image_tokens + num_action_tokens

        logits = jnp.reshape(logits, (bs, seqlen, time_step_tokens, self.model.vocab_size))
        logits = logits[:, -1, num_image_tokens - 1 : -1] # [bs, num_action_tokens, vocab_size]
        values = values[:, -1, :]  # [bs, 1]

        outputs["batch_stats"] = new_variables.get("batch_stats", params["batch_stats"])        
        return logits, values, outputs

    def _rollout(self, params, observation, rng):
        logits, values, outputs = self._forward(
            params=params,
            observation=observation,
            rng=rng,
            train=False
        )

        # only care about action tokens [1:8]
        clipped_logits = logits[:, 1:8] # [bs, 7, vocab_size]

        # Sample actions from the logits
        action_token = jax.random.categorical(rng, logits, axis=-1)
        clipped_action_tokens = action_token[:, 1:8]  # [bs, 7]

        log_probs_all = jax.nn.log_softmax(clipped_logits, axis=-1)
        idx = clipped_action_tokens[..., jnp.newaxis]  # [bs, 7, 1]
        log_probs_tokens = jnp.take_along_axis(log_probs_all, idx, axis=-1)[..., 0]  # [bs, 7]
        log_prob = jnp.sum(log_probs_tokens, axis=-1, keepdims=True)         # [bs, 1]

        actions_dict = detokenize_action(
            action_token, self.model.vocab_size, self.model.world_vector_range)
        
        # only select the world vector, rotation delta, gripper
        actions = jnp.concatenate(
            [
                actions_dict["world_vector"],
                actions_dict["rotation_delta"],
                actions_dict["gripper_closedness_action"],
            ],
            axis=-1,
        )
        return actions, log_prob, values, outputs
    
    def _update(self, params, observation, act_tokens, rng):
        logits, values, outputs = self._forward(
            params=params,
            observation=observation,
            rng=rng,
            train=True
        )

        clipped_logits = logits[:, 1:8] # [bs, 7, vocab_size]
        log_probs_all = jax.nn.log_softmax(clipped_logits, axis=-1)          # [bs, 7, vocab]
        idx = act_tokens[:, 1:8][..., None]                                        # [bs, 7, 1]
        log_probs_tokens = jnp.take_along_axis(log_probs_all, idx, axis=-1)[..., 0]  # [bs, 7]
        log_prob = jnp.sum(log_probs_tokens, axis=-1, keepdims=True)         # [bs, 1]

        return log_prob, values, outputs

    # should never be jit compiled (not pure)
    def act(self, inputs, role="", params=None):
        images = self._process_inputs_by_role(role, inputs)
        bs = images.shape[0]
        embeds = jnp.repeat(self.embeds, bs, axis=0)
        observation = {"image": images, "natural_language_embedding": embeds}

        params = self.state_dict.params if params is None else params
        params["batch_stats"] = self.batch_stats

        self.rng, call_rng = jax.random.split(self.rng)
        actions, log_prob, values, outputs = self._jitted_rollout(
            params=params,
            observation=observation,
            rng=call_rng,
        )

        actions = _rescale_action_with_bound(
            actions,
            low=-2.0,
            high=2.0,
            post_scaling_min=self._a2c_clip_actions_min,
            post_scaling_max=self._a2c_clip_actions_max,
        )

        if self._a2c_reduction:
            log_prob = self._a2c_reduction(log_prob, axis=-1)
        if log_prob.ndim != actions.ndim:
            log_prob = jnp.expand_dims(log_prob, -1)

        if self._d_clip_values:
            values = jnp.clip(values, a_min=self._d_clip_values_min, a_max=self._d_clip_values_max)

        outputs["values"] = values
        outputs["stddev"] = jnp.zeros_like(values)

        return actions, log_prob, outputs

    # update method should handle formatting params/inputs (since it's jitted)
    def update_act(self, inputs, role="", params=None, rng=None):
        images = self._process_inputs_by_role(role, inputs) 
        bs = images.shape[0]
        embeds = jnp.repeat(self.embeds, bs, axis=0)
        observation = {"image": images, "natural_language_embedding": embeds}

        actions = inputs["actions"]
        actions_dict = {
            "world_vector": actions[:, :3],
            "rotation_delta": actions[:, 3:6],
            "gripper_closedness_action": actions[:, 6:7],
            "base_displacement_vertical_rotation": jnp.zeros((bs, 1)),
            "base_displacement_vector": jnp.zeros((bs, 2)),
            "terminate_episode": jnp.zeros((bs, 1)),
        }
        act_tokens = tokenize_action(
            actions=actions_dict,
            vocab_size=self.model.vocab_size,
            world_vector_range=self.model.world_vector_range,
        )

        rng, call_rng = jax.random.split(rng)
        log_prob, values, outputs = self._jitted_update(
            params=params,
            observation=observation,
            act_tokens=act_tokens,
            rng=call_rng,
        )

        if self._a2c_reduction:
            log_prob = self._a2c_reduction(log_prob, axis=-1)
        if log_prob.ndim != actions.ndim:
            log_prob = jnp.expand_dims(log_prob, -1)

        if self._d_clip_values:
            values = jnp.clip(values, a_min=self._d_clip_values_min, a_max=self._d_clip_values_max)

        outputs["values"] = values
        outputs["stddev"] = jnp.zeros_like(values)
        outputs["new_rng"] = rng
        return log_prob, outputs

    def init_state_dict(
        self, 
        role: str, 
        inputs: Mapping[str, Union[np.ndarray, jax.Array]] = {}, 
        key: Optional[jax.Array] = None,
        ckpt: Optional[str] = None
    ) -> None:
        """Initialize state dictionary

        :param role: Role play by the model
        :type role: str
        :param inputs: Model inputs. The most common keys are:

                        - ``"states"``: state of the environment used to make the decision
                        - ``"taken_actions"``: actions taken by the policy for the given states

                       If not specified, the keys will be populated with observation and action space samples
        :type inputs: dict of np.ndarray or jax.Array, optional
        :param key: Pseudo-random number generator (PRNG) key (default: ``None``).
                    If not provided, the skrl's PRNG key (``config.jax.key``) will be used
        :type key: jax.Array, optional
        """
        if ckpt:
            raise NotImplementedError("Loading from checkpoint is not implemented yet.")

        obs = {
            "image": jnp.ones((1, self.seqlen, 300, 300, 3)),
            "natural_language_embedding": jnp.ones((1, self.seqlen, self.model.num_action_tokens)),
        }

        if key is None:
            key = config.jax.key

        rngs = {
            "params": key, 
            "random": key
        }

        variables = self.model.init(
            rngs=rngs,
            obs=obs,
            act=None,
            act_tokens=jnp.zeros((1, 6, 11)),
            train=False
        )

        from flax.training import checkpoints
        checkpoint_path = "improve/custom/models/rt1/rtx_ckpt/"
        state_dict = checkpoints.restore_checkpoint(checkpoint_path, None)

        self.batch_stats = state_dict['batch_stats']
        params = variables['params']

        # replace parts of the params with the checkpoint params
        for module_name, module_params in state_dict['params'].items():
            if module_name in params:
                params[module_name] = module_params

        # init internal state dict
        with jax.default_device(self.device):
            self.state_dict = StateDict.create(apply_fn=None, params={"params": params})
    
    ### From jax base skrl
    def set_mode(self, mode: str) -> None:
        """Set the model mode (training or evaluation)

        :param mode: Mode: ``"train"`` for training or ``"eval"`` for evaluation
        :type mode: str

        :raises ValueError: If the mode is not ``"train"`` or ``"eval"``
        """
        if mode == "train":
            self.training = True
        elif mode == "eval":
            self.training = False
        else:
            raise ValueError("Invalid mode. Use 'train' for training or 'eval' for evaluation")

    def _load_embeddings(self, task):
        self._load_cache()
        if task in self._embed_cache:
            print(f"Using cached embedding for task: '{task}'")
            embeds = jnp.array(self._embed_cache[task])
        else:
            print(f"Computing new embedding for task: '{task}'")
            self.llm = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
            embeds = self.llm([task]).numpy()
            
            self._embed_cache[task] = embeds.tolist()
            self._save_cache()
            
            del self.llm
            import gc
            gc.collect()

        self.embeds = jnp.array(embeds)
        self.embeds = jnp.expand_dims(self.embeds, 1)
        self.embeds = jnp.repeat(self.embeds, self.seqlen, axis=1) # leave batch for later
    
    def _load_cache(self):
        """Load cache from external JSON file."""
        self._embed_cache = {}
        self._cache_file = "task_embeddings.json"

        if os.path.exists(self._cache_file):
            try:
                with open(self._cache_file, 'r') as f:
                    self._embed_cache = json.load(f)
                print(f"✓ Loaded {len(self._embed_cache)} cached embeddings from {self._cache_file}")
            except (json.JSONDecodeError, Exception) as e:
                print(f"⚠ Cache file corrupted ({e}), starting fresh")
                self._embed_cache = {}
        else:
            print(f"ℹ No cache file found at {self._cache_file}, starting fresh")
            self._embed_cache = {}
    
    def _save_cache(self):
        """Save cache to external JSON file."""
        try:
            with open(self._cache_file, 'w') as f:
                json.dump(self._embed_cache, f, indent=2)
            print(f"✓ Saved cache to {self._cache_file}")
        except Exception as e:
            print(f"✗ Failed to save cache: {e}")
    
    @classmethod 
    def clear_cache(cls):
        """Clear the embedding cache and delete the file."""
        cls._embed_cache = {}
        cls._cache_loaded = False
        if os.path.exists(cls._cache_file):
            os.remove(cls._cache_file)
            print(f"✓ Cache cleared and {cls._cache_file} deleted")
        else:
            print("ℹ No cache file to delete")
    
    @classmethod
    def get_cached_tasks(cls):
        """Get list of all cached tasks."""
        if not cls._cache_loaded:
            temp_instance = cls.__new__(cls)  # Create temp instance to load cache
            temp_instance._load_cache()
            cls._cache_loaded = True
        return list(cls._embed_cache.keys())

    def get_entropy(self, stddev: jax.Array, role: str = "") -> jax.Array:
        """Compute and return the entropy of the model

        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        :return: Entropy of the model
        :rtype: jax.Array
        """
        return _entropy(stddev)

    def __getattr__(self, name):
        # use fallback to return attribute from model
        if name in self.__dict__:
            return self.__dict__[name]
        return getattr(self.model, name)
    
@jax.jit
def _entropy(logits):
    logits = logits - \
        jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    logits = logits.clip(min=jnp.finfo(logits.dtype).min)
    p_log_p = logits * jax.nn.softmax(logits)
    return -p_log_p.sum(-1)

def _rescale_action_with_bound(
    actions: jax.Array,
    low: float,
    high: float,
    safety_margin: float = 0.0,
    post_scaling_max: float = 1.0,
    post_scaling_min: float = -1.0,
) -> np.ndarray:
    """Formula taken from https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range."""
    resc_actions = (actions - low) / (high - low) * (
        post_scaling_max - post_scaling_min
    ) + post_scaling_min
    return jnp.clip(
        resc_actions,
        post_scaling_min + safety_margin,
        post_scaling_max - safety_margin,
    )