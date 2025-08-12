from improve.custom.env.isaacgym_envs import _torch2jax
import jax.numpy as jnp
from skrl.utils.spaces.torch import tensorize_space

IMAGE_SHAPE = (256, 256, 3)

class TorchToJaxEnvWrapper:
    def __init__(self, env):
        self._env = env
    
    def reset(self, *args, **kwargs):
        obs, info = self._env.reset(*args, **kwargs)
        obs = jnp.array(obs["obs"].cpu(), dtype=jnp.float32)
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        obs = jnp.array(obs["obs"].cpu(), dtype=jnp.float32)
        reward = jnp.array(reward, dtype=jnp.float32)
        terminated = jnp.array(terminated, dtype=jnp.bool_)
        truncated = jnp.array(truncated, dtype=jnp.bool_)

        return obs, reward, terminated, truncated, info
    
    def render(self):
        return jnp.array(self._env.render(), dtype=jnp.float32)
    
    def __getattr__(self, name):
        return getattr(self._env, name)

class SelectObsWrapper:
    def __init__(self, env):
        self._env = env
        self.use_camera_obs = getattr(env, "use_camera_obs", False)

        self._raw_obs = None

    def reset(self, *args, **kwargs):
        obs, info = self._env.reset(*args, **kwargs)
        self._raw_obs = obs["image"]

        # Select appropriate observation
        selected_obs = obs.get("image") if self.use_camera_obs else obs.get("vector")

        return selected_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._raw_obs = obs["image"]

        # Select appropriate observation
        selected_obs = obs.get("image") if self.use_camera_obs else obs.get("vector")

        return selected_obs, reward, terminated, truncated, info

    def render(self):
        return self._raw_obs.reshape(-1, *IMAGE_SHAPE)

    @property
    def observation_space(self):
        if self.use_camera_obs:
            return self._env.observation_space["image"]
        else:
            return self._env.observation_space["vector"]

    # Forward all other attributes and methods
    def __getattr__(self, name):
        return getattr(self._env, name)

class CustomIsaacGymEnvWrapper:
    def __init__(self, env):
        self._env = env
        self._raw_obs = None

    def _get_observation(self, observations):
        """Get the observation from the environment"""
        if "image" in observations:
            image = observations["image"].cpu().numpy()
        else:
            image = jnp.zeros((1, *IMAGE_SHAPE), dtype=jnp.float32)

        obs = _torch2jax(tensorize_space(self.observation_space, observations["obs"]))
        
        return {
            "obs": obs,
            "image": image
        }

    def reset(self, *args, **kwargs):
        obs, info = self._env.reset(*args, **kwargs)
        obs = self._get_observation(obs)

        self._raw_obs = obs["image"]
        selected_obs = obs["obs"]

        return selected_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        obs = self._get_observation(obs)
        self._raw_obs = obs["image"]
        selected_obs = obs["obs"]

        return selected_obs, reward, terminated, truncated, info

    def render(self):
        return self._raw_obs.reshape(-1, *IMAGE_SHAPE)

    def __getattr__(self, name):
        return getattr(self._env, name)