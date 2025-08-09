import numpy as np
import gymnasium as gym
import jax.numpy as jnp

IMAGE_SHAPE = (256, 256, 3)

class RandomObsOneRewardOneTimestepEnvWrapper:
    """A wrapper that provides random observations, one reward, and one timestep."""
    def __init__(self, env):
        self._env = env
        self.use_camera_obs = getattr(env, "use_camera_obs", False)

    def reset(self, *args, **kwargs):
        obs, info = self._env.reset(*args, **kwargs)
        selected_obs = obs.get("image") if self.use_camera_obs else obs.get("vector")

        selected_obs = np.random.normal(size=selected_obs.shape)
        return selected_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        selected_obs = obs.get("image") if self.use_camera_obs else obs.get("vector")

        selected_obs = np.random.normal(selected_obs)
        reward = np.ones_like(reward)
        terminated = np.ones_like(terminated)
        truncated = np.zeros_like(truncated)
        return selected_obs, reward, terminated, truncated, info

    def render(self):
        raise NotImplementedError("Rendering is not implemented in Debug Envs")

    @property
    def observation_space(self):
        if self.use_camera_obs:
            return self._env.observation_space["image"]
        else:
            return self._env.observation_space["vector"]

    # Forward all other attributes and methods
    def __getattr__(self, name):
        return getattr(self._env, name)
    
class TwoObsTwoRewardOneTimestepEnvWrapper:
    """A wrapper that provides two observations, two rewards, and one timestep."""
    def __init__(self, env):
        self._env = env
        self.use_camera_obs = getattr(env, "use_camera_obs", False)
        self.prev_sign = 1

    def reset(self, *args, **kwargs):
        obs, info = self._env.reset(*args, **kwargs)
        selected_obs = obs.get("image") if self.use_camera_obs else obs.get("vector")

        self.sign = np.random.choice([1, -1])
        selected_obs = np.ones_like(selected_obs) * self.sign
        return selected_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        selected_obs = obs.get("image") if self.use_camera_obs else obs.get("vector")

        reward = np.ones_like(reward) * self.sign
        terminated = np.ones_like(terminated)
        truncated = np.zeros_like(truncated)

        self.sign = np.random.choice([1, -1])
        selected_obs = np.ones_like(selected_obs) * self.sign
        return selected_obs, reward, terminated, truncated, info

    def render(self):
        raise NotImplementedError("Rendering is not implemented in Debug Envs")

    @property
    def observation_space(self):
        if self.use_camera_obs:
            return self._env.observation_space["image"]
        else:
            return self._env.observation_space["vector"]

    # Forward all other attributes and methods
    def __getattr__(self, name):
        return getattr(self._env, name)
    
class TwoObsOneRewardTwoTimestepEnvWrapper:
    """A wrapper that provides two observations, one reward, and two timesteps."""
    def __init__(self, env):
        self._env = env
        self.use_camera_obs = getattr(env, "use_camera_obs", False)
        self.prev_sign = 1
        self.timestep = 0

    def reset(self, *args, **kwargs):
        obs, info = self._env.reset(*args, **kwargs)
        selected_obs = obs.get("image") if self.use_camera_obs else obs.get("vector")

        selected_obs = -np.ones_like(selected_obs)
        self.timestep = 0
        return selected_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        selected_obs = obs.get("image") if self.use_camera_obs else obs.get("vector")

        if self.timestep % 2 == 0:
            selected_obs = -np.ones_like(selected_obs)
            reward = np.zeros_like(reward)
            terminated = np.zeros_like(terminated)
        else:
            selected_obs = np.ones_like(selected_obs)
            reward = np.ones_like(reward)
            terminated = np.ones_like(terminated)
        truncated = np.zeros_like(truncated)
        self.timestep += 1
        
        return selected_obs, reward, terminated, truncated, info

    def render(self):
        raise NotImplementedError("Rendering is not implemented in Debug Envs")

    @property
    def observation_space(self):
        if self.use_camera_obs:
            return self._env.observation_space["image"]
        else:
            return self._env.observation_space["vector"]

    # Forward all other attributes and methods
    def __getattr__(self, name):
        return getattr(self._env, name)

# only policy
class OneObsTwoActionTwoRewardOneTimestepEnvWrapper:
    """A wrapper that provides one observation, two actions, two rewards, and one timestep."""
    def __init__(self, env):
        self._env = env
        self.use_camera_obs = getattr(env, "use_camera_obs", False)

    def reset(self, *args, **kwargs):
        obs, info = self._env.reset(*args, **kwargs)
        selected_obs = obs.get("image") if self.use_camera_obs else obs.get("vector")

        selected_obs = np.ones_like(selected_obs)
        return selected_obs, info

    def step(self, action):
        n_envs = self._env.num_envs
        action_shape = self._env.action_space.shape 

        dummy_action = jnp.zeros((n_envs, *action_shape))
        obs, reward, terminated, truncated, info = self._env.step(dummy_action)
        selected_obs = obs.get("image") if self.use_camera_obs else obs.get("vector")

        terminated = np.ones_like(terminated)
        truncated = np.zeros_like(truncated)
        selected_obs = np.ones_like(selected_obs)

        reward = np.where(np.all(action > 0, axis=1), 1.0, -1.0).reshape(-1, 1)
        return selected_obs, reward, terminated, truncated, info

    def render(self):
        raise NotImplementedError("Rendering is not implemented in Debug Envs")

    @property
    def observation_space(self):
        if self.use_camera_obs:
            return self._env.observation_space["image"]
        else:
            return self._env.observation_space["vector"]
        
    @property
    def action_space(self):
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
    def __getattr__(self, name):
        return getattr(self._env, name)
    
    
class TwoObsTwoActionTwoRewardOneTimestepEnvWrapper:
    """A wrapper that provides two observations, two actions, two rewards, and one timestep."""
    def __init__(self, env):
        self._env = env
        self.use_camera_obs = getattr(env, "use_camera_obs", False)
        self.current_obs = None
    
    def reset(self, *args, **kwargs):
        obs, info = self._env.reset(*args, **kwargs)
        selected_obs = obs.get("image") if self.use_camera_obs else obs.get("vector")
        
        # Generate random +1/-1 observations for each environment
        n_envs = self._env.num_envs
        obs_shape = selected_obs.shape[1:]  # Remove batch dimension
        
        # Random +1 or -1 for each environment
        random_signs = np.random.choice([-1, 1], size=(n_envs,))
        selected_obs = np.broadcast_to(random_signs[:, None], (n_envs, *obs_shape)).astype(np.float32)
        
        self.current_obs = selected_obs
        return selected_obs, info
    
    def step(self, action):
        n_envs = self._env.num_envs
        action_shape = self._env.action_space.shape 
        dummy_action = jnp.zeros((n_envs, *action_shape))
        obs, _, terminated, truncated, info = self._env.step(dummy_action)
        
        selected_obs = obs.get("image") if self.use_camera_obs else obs.get("vector")
        obs_shape = selected_obs.shape[1:]
        random_signs = np.random.choice([-1, 1], size=(n_envs,))
        selected_obs = np.broadcast_to(random_signs[:, None], (n_envs, *obs_shape)).astype(np.float32)
        
        terminated = np.ones_like(terminated)
        truncated = np.zeros_like(truncated)
        
        obs_signs = np.sign(self.current_obs[:, 0])
        action_signs = np.sign(action[:, 0])
        
        reward = np.where(obs_signs == action_signs, 1.0, -1.0).reshape(-1, 1)
        self.current_obs = selected_obs
        return selected_obs, reward, terminated, truncated, info
    
    def render(self):
        raise NotImplementedError("Rendering is not implemented in Debug Envs")
    
    @property
    def observation_space(self):
        if self.use_camera_obs:
            return self._env.observation_space["image"]
        else:
            return self._env.observation_space["vector"]
        
    @property
    def action_space(self):
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
    def __getattr__(self, name):
        return getattr(self._env, name)
