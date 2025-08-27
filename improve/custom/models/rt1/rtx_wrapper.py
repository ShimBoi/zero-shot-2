from collections import deque
import copy
import flax
import gymnasium
from improve.custom.models.rt1.rt1 import RT1
from improve.custom.models.rt1.utils import detokenize_action
import jax
import jax.numpy as jnp
from skrl import config
import tensorflow as tf
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

    def __init__(self, observation_space, action_space, device, task="", clip_actions=False, clip_values=False, reduction="sum"):
        self._embed_cache = {}
        self._cache_file = "task_embeddings.json"

        self.rng = config.jax.key

        model = RT1(
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            task=task,
            clip_actions=clip_actions,
            clip_values=clip_values,
            reduction=reduction,
        )
        self.model = model

        self.hist = deque(maxlen=self.seqlen)
        self.num_image_history = 0

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

        self._jitted_apply = jax.jit(model.act, static_argnames=("role"))


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
            images = jax.image.resize(images, (images.shape[0], images.shape[1], 300, 300, 3), method='linear') / 255.0
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
        print("Max image value:", jnp.max(image), "Min image value:", jnp.min(image))
        return image

    def act(self, inputs, role="", params=None):
        images = self._process_inputs_by_role(role, inputs) 
        bs = images.shape[0]
        embeds = jnp.repeat(self.embeds, bs, axis=0)
        observation = {"image": images, "natural_language_embedding": embeds}

        params = copy.deepcopy(self.state_dict.params if params is None else params)
        params["batch_stats"] = self.batch_stats

        actions, log_prob, outputs = self._jitted_apply(
            params=params,
            inputs=observation,
            role="",
            rng=self.rng
        )

        # update rngs and batch stats
        new_rng = jax.random.fold_in(self.rng, self.num_image_history)
        outputs["rng"] = new_rng

        return actions, log_prob, outputs
    
    def _load_cache(self):
        """Load cache from external JSON file."""
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
        
        from flax.training import checkpoints
        checkpoint_path = "improve/custom/models/rt1/rtx_ckpt/"
        state_dict = checkpoints.restore_checkpoint(checkpoint_path, None)

        obs = {
            "image": jnp.ones((1, 15, 300, 300, 3)),
            "natural_language_embedding": jnp.ones((1, 15, 512)),
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
            act_tokens=self.act_tokens,
            train=False
        )

        self.batch_stats = state_dict['batch_stats']
        params = variables['params']

        # replace parts of the params with the checkpoint params
        for module_name, module_params in state_dict['params'].items():
            if module_name in params:
                params[module_name] = module_params

        # init internal state dict
        with jax.default_device(self.device):
            self.state_dict = StateDict.create(apply_fn=None, params={"params": params})

    def get_entropy(self, stddev: jax.Array, role: str = "") -> jax.Array:
        """Compute and return the entropy of the model

        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        :return: Entropy of the model
        :rtype: jax.Array
        """
        return _entropy(stddev)

    def __getattr__(self, name):
        return getattr(self.model, name)
    
@jax.jit
def _entropy(logits):
    logits = logits - \
        jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    logits = logits.clip(min=jnp.finfo(logits.dtype).min)
    p_log_p = logits * jax.nn.softmax(logits)
    return -p_log_p.sum(-1)

