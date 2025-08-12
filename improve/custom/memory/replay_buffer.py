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

    def sample_all(self, names, sequence_length: int = 15,
               minibatches: int = 1, state_shape=(256, 256, 3),
               is_image_obs: bool = True, epochs: int = 1, 
               rng_key: Optional[random.PRNGKey] = None) -> Iterator[Dict[str, jnp.ndarray]]:
        """
        Optimized sampling method with epoch support using JAX operations.
        Pre-computes all samples once, then efficiently shuffles and batches for multiple epochs.
        
        Args:
            epochs: Number of times to iterate through the dataset with different shuffling
            rng_key: JAX random key for shuffling (will create one if None)
        
        Returns:
            Iterator yielding batched samples as JAX arrays
        """
        if rng_key is None:
            rng_key = random.PRNGKey(0)
        
        M, B = self.memory_size, self.num_envs
        
        if is_image_obs:
            return self._sample_image_obs_optimized(names, sequence_length, minibatches, 
                                                state_shape, epochs, rng_key)
        else:
            return self._sample_regular_obs_optimized(names, minibatches, epochs, rng_key)

    def _sample_image_obs_optimized(self, names, sequence_length, minibatches, 
                                state_shape, epochs, rng_key):
        """Optimized image observation sampling with vectorized JAX operations."""
        M, B = self.memory_size, self.num_envs
        H, W, C = state_shape
        flat = H * W * C
        
        # Validate shapes
        expected_shape = (flat,)
        actual_shape = self._buffers["states"].shape[2:]
        assert actual_shape == expected_shape, \
            f"Expected flattened states {expected_shape}, got {actual_shape}"
        
        # Pre-compute ALL samples once (this is the expensive part we want to do only once)
        def create_all_samples():
            """Create all samples for all environments and timesteps using JAX ops."""
            total_samples = M * B
            
            # Pre-allocate arrays
            all_states = jnp.zeros((total_samples, sequence_length, H, W, C), 
                                dtype=self._buffers["states"].dtype)
            all_other_features = {}
            
            # Pre-allocate other feature arrays
            for n in names:
                if n != "states":
                    feat_shape = self._buffers[n].shape[2:]
                    all_other_features[n] = jnp.zeros((total_samples,) + feat_shape,
                                                    dtype=self._buffers[n].dtype)
            
            # Build all samples
            sample_idx = 0
            states_buf = self._buffers["states"]  # (M, B, flat)
            
            # Vectorized approach for each environment
            for b in range(B):
                env_states = states_buf[:, b]  # (M, flat)
                
                # Create all sliding windows for this environment at once
                for t in range(M):
                    start = max(0, t - sequence_length + 1)
                    window = env_states[start:t+1]  # (L, flat)
                    
                    L = window.shape[0]
                    if L < sequence_length:
                        # Pad with zeros at the beginning
                        pad_length = sequence_length - L
                        padding = jnp.zeros((pad_length, flat), dtype=window.dtype)
                        window = jnp.concatenate([padding, window], axis=0)
                    
                    # Reshape to image format
                    window_reshaped = window.reshape(sequence_length, H, W, C)
                    all_states = all_states.at[sample_idx].set(window_reshaped)
                    
                    # Store other features for this timestep
                    for n in names:
                        if n != "states":
                            all_other_features[n] = all_other_features[n].at[sample_idx].set(
                                self._buffers[n][t, b])
                    
                    sample_idx += 1
            
            # Combine into final dict
            samples_dict = {"states": all_states}
            samples_dict.update(all_other_features)
            
            return samples_dict
        
        # THIS IS KEY: Create all samples ONCE, then reuse for all epochs
        print("Pre-computing all samples (this happens only once)...")
        all_samples = create_all_samples()
        total_samples = M * B
        print(f"Pre-computed {total_samples} samples")
        
        # Calculate batch size
        if minibatches > 1:
            batch_size = max(1, total_samples // minibatches)
        else:
            batch_size = total_samples
        
        def create_epoch_batches(epoch_rng_key):
            """Create shuffled batches for one epoch - this is fast since data is pre-computed."""
            # Shuffle indices
            indices = random.permutation(epoch_rng_key, jnp.arange(total_samples))
            
            # Create shuffled samples using advanced indexing (fast!)
            shuffled_samples = {}
            for n in names:
                shuffled_samples[n] = all_samples[n][indices]
            
            # Yield batches
            num_batches = (total_samples + batch_size - 1) // batch_size
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, total_samples)
                
                batch = {}
                for n in names:
                    batch[n] = shuffled_samples[n][start_idx:end_idx]
                
                yield batch
        
        # Generate batches for all epochs (each epoch has different shuffling)
        for epoch in range(epochs):
            if epoch > 0:
                print(f"Starting epoch {epoch + 1}/{epochs} (reusing pre-computed data)")
            epoch_rng_key, rng_key = random.split(rng_key)
            yield from create_epoch_batches(epoch_rng_key)

    def _sample_regular_obs_optimized(self, names, minibatches, epochs, rng_key):
        """Optimized regular observation sampling using JAX."""
        M, B = self.memory_size, self.num_envs
        total_samples = M * B
        
        # Pre-flatten all samples (do this once)
        samples_dict = {}
        for n in names:
            # Flatten first two dimensions: (M, B, ...) -> (M*B, ...)
            buf = self._buffers[n]
            samples_dict[n] = buf.reshape((total_samples,) + buf.shape[2:])
        
        # Calculate batch size
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
            for n in names:
                shuffled_samples[n] = samples_dict[n][indices]
            
            # Yield batches
            num_batches = (total_samples + batch_size - 1) // batch_size
            # print(f"Creating {num_batches} batches of size {batch_size} for epoch")
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, total_samples)
                
                batch = {}
                for n in names:
                    batch[n] = shuffled_samples[n][start_idx:end_idx]
                
                yield batch
        
        # Generate batches for all epochs
        for epoch in range(epochs):
            # print("Epoch:", epoch)
            epoch_rng_key, rng_key = random.split(rng_key)
            yield from create_epoch_batches(epoch_rng_key)

    def _sample_image_obs_vectorized_jax(self, names, sequence_length, minibatches, 
                                        state_shape, epochs, rng_key):
        """
        Ultra-optimized version using JAX's vmap and advanced indexing.
        This creates all sliding windows in a single vectorized operation.
        """
        M, B = self.memory_size, self.num_envs
        H, W, C = state_shape
        flat = H * W * C
        
        def create_all_samples_vectorized():
            """Create all samples using vectorized operations."""
            # Create index arrays for all samples
            env_indices = jnp.repeat(jnp.arange(B), M)  # [0,0,...,0, 1,1,...,1, ...]
            time_indices = jnp.tile(jnp.arange(M), B)   # [0,1,...,M-1, 0,1,...,M-1, ...]
            
            # Compute start indices for all windows
            start_indices = jnp.maximum(0, time_indices - sequence_length + 1)
            
            # Create sequence offset indices
            seq_offsets = jnp.arange(sequence_length)[None, :]  # (1, seq_len)
            window_indices = start_indices[:, None] + seq_offsets  # (total_samples, seq_len)
            
            # Clamp indices to valid range
            window_indices = jnp.clip(window_indices, 0, M - 1)
            
            # Create padding mask
            valid_mask = (start_indices[:, None] + seq_offsets) <= time_indices[:, None]
            
            # Gather all windows using advanced indexing
            states_buf = self._buffers["states"]  # (M, B, flat)
            
            # This is the magic: gather all windows at once
            # Use fancy indexing: states_buf[window_indices, env_indices[:, None]]
            total_samples = M * B
            gathered_windows = []
            
            for i in range(total_samples):
                env_idx = env_indices[i]
                window_idx = window_indices[i]
                window = states_buf[window_idx, env_idx]  # (seq_len, flat)
                
                # Apply padding mask
                window = jnp.where(valid_mask[i, :, None], window, 0)
                gathered_windows.append(window)
            
            all_states = jnp.stack(gathered_windows)  # (total_samples, seq_len, flat)
            all_states = all_states.reshape(total_samples, sequence_length, H, W, C)
            
            # Handle other features
            samples_dict = {"states": all_states}
            for n in names:
                if n != "states":
                    feat_data = self._buffers[n][time_indices, env_indices]
                    samples_dict[n] = feat_data
            
            return samples_dict
        
        # Pre-compute all samples
        print("Pre-computing all samples with vectorized operations...")
        all_samples = create_all_samples_vectorized()
        total_samples = M * B
        print(f"Pre-computed {total_samples} samples")
        
        # Rest is the same as before...
        batch_size = max(1, total_samples // minibatches) if minibatches > 1 else total_samples
        
        def create_epoch_batches(epoch_rng_key):
            indices = random.permutation(epoch_rng_key, jnp.arange(total_samples))
            shuffled_samples = {n: all_samples[n][indices] for n in names}
            
            num_batches = (total_samples + batch_size - 1) // batch_size
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, total_samples)
                yield {n: shuffled_samples[n][start_idx:end_idx] for n in names}
        
        for epoch in range(epochs):
            if epoch > 0:
                print(f"Starting epoch {epoch + 1}/{epochs} (reusing pre-computed data)")
            epoch_rng_key, rng_key = random.split(rng_key)
            yield from create_epoch_batches(epoch_rng_key)