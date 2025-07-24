IMAGE_SHAPE = (256, 256, 3)

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