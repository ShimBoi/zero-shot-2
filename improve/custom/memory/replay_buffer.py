import numpy as np
import tensorflow as tf
from gym.spaces import Space, Box

from typing import Union, Tuple

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
        if size > 1:
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

            if arr.ndim == 2 and arr.shape[-1] == 1:
                # convert (num_envs, 1) to (num_envs,)
                arr = arr.squeeze(-1)
            self._buffers[name][idx] = arr
        self._pos += 1

    def sample_all(self, names, sequence_length: int = 15,
                   minibatches: int = 1, state_shape=(256, 256, 3),
                   is_image_obs: bool = True) -> tf.data.Dataset:
        M, B = self.memory_size, self.num_envs

        # precompute feature‐shapes and dtypes
        if is_image_obs:
            feat_shapes = {n: self._buffers[n].shape[2:] for n in names}
            dtypes      = {n: tf.as_dtype(self._buffers[n].dtype) for n in names}

            def gen():
                H, W, C = state_shape
                flat = H * W * C
                assert feat_shapes["states"] == (flat,), \
                    f"expected flattened feat_states {flat}, got {feat_shapes['states']}"

                for b in range(B):
                    # grab this env’s entire timeseries for each key
                    env_bufs = {n: self._buffers[n][:, b]  # (M, *feat_n)
                                for n in names}

                    states_buf = env_bufs["states"]       # shape (M, flat)
                    for t in range(M):
                        # make the sliding window of length T on states
                        start = max(0, t - sequence_length + 1)
                        window = states_buf[start : t + 1]   # (L, flat)
                        L = window.shape[0]
                        if L < sequence_length:
                            pad = sequence_length - L
                            zeros = np.zeros((pad, flat), dtype=window.dtype)
                            window = np.concatenate([zeros, window], axis=0)
                        # now window.shape == (T, flat)

                        # **here** reshape it back into images
                        window = window.reshape((sequence_length, H, W, C))

                        sample = {"states": window}  # (T, H, W, C)
                        # the other keys stay 1-per-timestep
                        for n in names:
                            if n == "states":
                                continue
                            sample[n] = env_bufs[n][t]  # shape (*feat_n)

                        yield sample

            # tf.Dataset spec
            output_types  = {n: dtypes[n] for n in names}
            output_shapes = {
                "states": (sequence_length, *state_shape),
                **{n: feat_shapes[n] for n in names if n != "states"}
            }

            ds = tf.data.Dataset.from_generator(
                gen,
                output_types=output_types,
                output_shapes=output_shapes,
            )
            ds = ds.shuffle(buffer_size=M * B)
            ds = ds.batch(minibatches, drop_remainder=False)
            return ds
        else:
            M, B = self.memory_size, self.num_envs

            def gen():
                for m in range(M):
                    for b in range(B):
                        sample = {n: self._buffers[n][m, b] for n in names}
                        yield sample

            output_types = {n: tf.as_dtype(self._buffers[n].dtype) for n in names}
            output_shapes = {n: self._buffers[n].shape[2:] for n in names}

            dataset = tf.data.Dataset.from_generator(
                gen,
                output_types=output_types,
                output_shapes=output_shapes,
            )

            total_samples = M * B
            batch_size = total_samples // minibatches

            dataset = dataset.shuffle(buffer_size=total_samples)
            dataset = dataset.batch(batch_size, drop_remainder=False)
            return dataset