import tensorflow as tf
from gym.spaces import Space, Box
import jax.numpy as jnp
import numpy as np
from jax import random
from typing import Dict, Iterator, Optional, Union, Tuple

class ReplayBuffer:
    """
    A simple replay buffer that stores transitions and produces
    a tf.data.Dataset of length-T windows ready for RT-1 training.

    Each add_sample call writes one timestep for all envs.
    get_dataset builds sliding windows of size sequence_length,
    batches them with batch_size, and prefetches.
    """
    def __init__(self, memory_size: int, num_envs: int):
        self.memory_size = memory_size
        self.num_envs    = num_envs
        self._pos        = 0
        self._buffers    = {}  # name -> np.ndarray of shape (M, B, *feat)

    def create_tensor(self, name: str, size: Union[int, Tuple[int,...]], dtype):
        shape = (self.memory_size, self.num_envs)
        shape += (size,)

        self._buffers[name] = np.zeros(shape,dtype=dtype)

    def get_tensor_by_name(self, name: str, keepdim: bool = True) -> np.ndarray:
        """
        Return the full buffer for `name`.

        If keepdim:
          shape = (memory_size, num_envs, *feat)
        else:
          shape = (memory_size * num_envs, *feat)
        """
        if name not in self._buffers:
            raise KeyError(f"No tensor named {name}")
        arr = self._buffers[name]
        if keepdim:
            return arr
        # flatten time × env dims
        M, B = arr.shape[0], arr.shape[1]
        feat = arr.shape[2:]
        return arr.reshape((M * B,) + feat)

    def set_tensor_by_name(self, name: str, tensor: np.ndarray) -> None:
        """
        Overwrite the entire buffer for `name` with `tensor`.
        `tensor` must have shape either
          (memory_size, num_envs, *feat)
        or
          (memory_size * num_envs, *feat)
        """
        if name not in self._buffers:
            raise KeyError(f"No tensor named {name}")

        buf = self._buffers[name]
        # same fully-shaped
        if tensor.shape == buf.shape:
            np.copyto(buf, tensor)
            return

        # maybe flattened first two dims?
        M, B = buf.shape[0], buf.shape[1]
        feat = buf.shape[2:]
        if tensor.shape == (M * B,) + feat:
            tensor = tensor.reshape((M, B) + feat)
            np.copyto(buf, tensor)
            return

        raise ValueError(
            f"Cannot set tensor {name}: expected shape {buf.shape} "
            f"or {(M*B,) + feat}, got {tensor.shape}"
        )

    def add_samples(self, **kwargs):
        """
        kwargs[name] is an array of shape (num_envs, *feat_shape)
        """
        idx = self._pos % self.memory_size
        for name, arr in kwargs.items():
            if name not in self._buffers:
                continue

            self._buffers[name][idx] = arr
        self._pos += 1

    def sample_all(
        self, 
        names, 
        sequence_length: int = 15,
        minibatches: int = 1, 
        is_image_obs: bool = True, 
        epochs: int = 1,
        rng_key: Optional[random.PRNGKey] = None) -> Iterator[Dict[str, jnp.ndarray]]:
        """
        Optimized sampling method with epoch support using JAX operations.
        
        Args:
            names: List of tensor names to include in samples
            sequence_length: Length of observation sequences (only used when is_image_obs=True)
            minibatches: Number of minibatches to split data into
            is_image_obs: Whether to create sliding windows for image observations
            epochs: Number of times to iterate through the dataset with different shuffling
            rng_key: JAX random key for shuffling (will create one if None)
        
        Returns:
            Iterator yielding batched samples as JAX arrays
        """
        if rng_key is None:
            rng_key = random.PRNGKey(0)
        
        M, B = self.memory_size, self.num_envs
        
        if is_image_obs:
            return self._sample_image_obs_windowed(names, sequence_length, minibatches, epochs, rng_key)
        else:
            return self._sample_regular_obs_optimized(names, minibatches, epochs, rng_key)

    def _sample_image_obs_windowed(self, names, sequence_length, minibatches, epochs, rng_key):
        """
        Create sliding windows of length sequence_length for image observations on-demand.
        Memory-efficient version that doesn't pre-compute all samples.
        """
        M, B = self.memory_size, self.num_envs
        total_samples = M * B
        
        # Calculate image dimensions from flattened states
        flat_size = self._buffers["states"].shape[2:][0]  # Should be H*W*C
        assert flat_size % 3 == 0, f"Flattened size {flat_size} not divisible by 3 (RGB channels)"
        hw = flat_size // 3
        H = W = int(np.sqrt(hw))
        assert H * W * 3 == flat_size, f"Cannot reshape {flat_size} into square image"
        C = 3
        
        # Calculate batch size for minibatches
        if minibatches > 1:
            batch_size = max(1, total_samples // minibatches)
        else:
            batch_size = total_samples
        
        def create_sample_on_demand(sample_idx):
            """Create a single windowed sample on demand."""
            # Convert flat index to (env, timestep)
            env_idx = sample_idx // M
            t = sample_idx % M
            
            # Create windowed states
            start_idx = t - sequence_length + 1
            windowed_states = np.zeros((sequence_length, H, W, C), dtype=self._buffers["states"].dtype)
            
            for seq_pos in range(sequence_length):
                actual_t = start_idx + seq_pos
                if actual_t >= 0:  # Skip padding (already zeros)
                    flat_state = self._buffers["states"][actual_t, env_idx]
                    windowed_states[seq_pos] = flat_state.reshape(H, W, C)
            
            # Create sample dict
            sample = {"states": windowed_states}
            
            # Add other features at timestep t
            for name in names:
                if name != "states":
                    sample[name] = self._buffers[name][t, env_idx]
            
            return sample
        
        def create_epoch_batches(epoch_rng_key):
            """Create shuffled batches for one epoch, generating samples on demand."""
            # Shuffle sample indices
            indices = random.permutation(epoch_rng_key, jnp.arange(total_samples))
            indices_np = np.array(indices)
            
            # Yield batches
            num_batches = (total_samples + batch_size - 1) // batch_size
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, total_samples)
                
                # Get batch indices
                batch_indices = indices_np[start_idx:end_idx]
                batch_len = len(batch_indices)
                
                # Create batch arrays
                batch = {
                    "states": np.zeros((batch_len, sequence_length, H, W, C), 
                                    dtype=self._buffers["states"].dtype)
                }
                
                # Pre-allocate other feature arrays
                for name in names:
                    if name != "states":
                        feat_shape = self._buffers[name].shape[2:]
                        batch[name] = np.zeros((batch_len,) + feat_shape, 
                                            dtype=self._buffers[name].dtype)
                
                # Fill batch by creating samples on demand
                for batch_pos, sample_idx in enumerate(batch_indices):
                    sample = create_sample_on_demand(sample_idx)
                    
                    batch["states"][batch_pos] = sample["states"]
                    for name in names:
                        if name != "states":
                            batch[name][batch_pos] = sample[name]
                
                # Convert batch to JAX arrays
                jax_batch = {}
                for name in names:
                    jax_batch[name] = jnp.array(batch[name])
                
                yield jax_batch
        
        print(f"Will generate {total_samples} windowed samples on-demand with {minibatches} minibatches")
        
        # Generate batches for all epochs
        for epoch in range(epochs):
            if epoch > 0:
                print(f"Starting epoch {epoch + 1}/{epochs}")
            epoch_rng_key, rng_key = random.split(rng_key)
            yield from create_epoch_batches(epoch_rng_key)

    def _sample_regular_obs_optimized(self, names, minibatches, epochs, rng_key):
        """Optimized regular observation sampling using JAX (when sequence_length=1)."""
        M, B = self.memory_size, self.num_envs
        total_samples = M * B
        
        # Pre-flatten all samples: (M, B, ...) -> (M*B, ...)
        samples_dict = {}
        for name in names:
            buf = self._buffers[name]  # Shape: (M, B, *feature_shape)
            # Flatten first two dimensions
            flattened = buf.reshape((total_samples,) + buf.shape[2:])
            samples_dict[name] = jnp.array(flattened)
        
        # Calculate batch size for minibatches
        if minibatches > 1:
            batch_size = max(1, total_samples // minibatches)
        else:
            batch_size = total_samples
        
        def create_epoch_batches(epoch_rng_key):
            """Create shuffled batches for one epoch."""
            # Shuffle indices
            indices = random.permutation(epoch_rng_key, jnp.arange(total_samples))
            
            # Create shuffled samples
            shuffled_samples = {}
            for name in names:
                shuffled_samples[name] = samples_dict[name][indices]
            
            # Yield batches
            num_batches = (total_samples + batch_size - 1) // batch_size
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, total_samples)
                
                batch = {}
                for name in names:
                    batch[name] = shuffled_samples[name][start_idx:end_idx]
                
                yield batch
        
        # Generate batches for all epochs
        for epoch in range(epochs):
            epoch_rng_key, rng_key = random.split(rng_key)
            yield from create_epoch_batches(epoch_rng_key)