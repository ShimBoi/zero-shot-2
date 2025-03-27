from skrl import config
import jax
import jax.numpy as jnp
import copy


class RTXPreprocessor:
    """
    Preprocessor for the RTX environment.
    """

    def __init__(self, batch_size, seqlen, device):
        self.seqlen = seqlen
        self.device = config.jax.parse_device(device)
        self.hist = jnp.zeros((batch_size, seqlen, *(300, 300, 3)), dtype=jnp.float32)
        self.num_image_history = 0 

    def _add_to_history(self, image) -> None:
        self.hist = jnp.roll(self.hist, shift=-1, axis=1)
        self.hist = self.hist.at[:, -1].set(image)
        self.num_image_history = min(self.num_image_history + 1, self.seqlen)

    def _obtain_history(self):
        return self.hist

    def __call__(self, states, train=True, inverse=False):
        image = copy.deepcopy(states)
        image = jax.image.resize(image, (image.shape[0], 300, 300, 3), method='bilinear') / 225.0

        self._add_to_history(image)
        return self._obtain_history()