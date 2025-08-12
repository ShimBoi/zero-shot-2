from typing import List, Optional, Union

import os
import atexit
import contextlib
import sys
import tqdm

import jax.numpy as jnp
import numpy as np
from skrl import config, logger
from skrl.agents.jax import Agent
from skrl.envs.wrappers.jax import Wrapper
import cv2


def generate_equally_spaced_scopes(num_envs: int, num_simultaneous_agents: int) -> List[int]:
    """Generate a list of equally spaced scopes for the agents

    :param num_envs: Number of environments
    :type num_envs: int
    :param num_simultaneous_agents: Number of simultaneous agents
    :type num_simultaneous_agents: int

    :raises ValueError: If the number of simultaneous agents is greater than the number of environments

    :return: List of equally spaced scopes
    :rtype: List[int]
    """
    scopes = [int(num_envs / num_simultaneous_agents)] * \
        num_simultaneous_agents
    if sum(scopes):
        scopes[-1] += num_envs - sum(scopes)
    else:
        raise ValueError(
            f"The number of simultaneous agents ({num_simultaneous_agents}) is greater than the number of environments ({num_envs})"
        )
    return scopes


class Trainer:
    def __init__(
        self,
        env: Wrapper,
        agents: Union[Agent, List[Agent]],
        agents_scope: Optional[List[int]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        """Base class for trainers

        :param env: Environment to train on
        :type env: skrl.envs.wrappers.jax.Wrapper
        :param agents: Agents to train
        :type agents: Union[Agent, List[Agent]]
        :param agents_scope: Number of environments for each agent to train on (default: ``None``)
        :type agents_scope: tuple or list of int, optional
        :param cfg: Configuration dictionary (default: ``None``)
        :type cfg: dict, optional
        """
        self.cfg = cfg if cfg is not None else {}
        self.env = env
        self.agents = agents
        self.agents_scope = agents_scope if agents_scope is not None else []

        # get configuration
        self.timesteps = self.cfg.get("timesteps", 0)
        self.headless = self.cfg.get("headless", False)
        self.disable_progressbar = self.cfg.get("disable_progressbar", False)
        self.close_environment_at_exit = self.cfg.get(
            "close_environment_at_exit", True)
        self.environment_info = self.cfg.get("environment_info", "episode")
        self.stochastic_evaluation = self.cfg.get(
            "stochastic_evaluation", False)
        self.output_folder = self.cfg.get("output_folder", "output")
        self._save_video = self.cfg.get("save_video", False)

        self.initial_timestep = 0

        # setup agents
        self.num_simultaneous_agents = 0
        self._setup_agents()

        # register environment closing if configured
        if self.close_environment_at_exit:

            @atexit.register
            def close_env():
                logger.info("Closing environment")
                self.env.close()
                logger.info("Environment closed")

        # update trainer configuration to avoid duplicated info/data in distributed runs
        if config.jax.is_distributed:
            if config.jax.rank:
                self.disable_progressbar = True

        self.images = []

    def __str__(self) -> str:
        """Generate a string representation of the trainer

        :return: Representation of the trainer as string
        :rtype: str
        """
        string = f"Trainer: {self}"
        string += f"\n  |-- Number of parallelizable environments: {self.env.num_envs}"
        string += f"\n  |-- Number of simultaneous agents: {self.num_simultaneous_agents}"
        string += "\n  |-- Agents and scopes:"
        if self.num_simultaneous_agents > 1:
            for agent, scope in zip(self.agents, self.agents_scope):
                string += f"\n  |     |-- agent: {type(agent)}"
                string += f"\n  |     |     |-- scope: {scope[1] - scope[0]} environments ({scope[0]}:{scope[1]})"
        else:
            string += f"\n  |     |-- agent: {type(self.agents)}"
            string += f"\n  |     |     |-- scope: {self.env.num_envs} environment(s)"
        return string

    def _setup_agents(self) -> None:
        """Setup agents for training

        :raises ValueError: Invalid setup
        """
        # validate agents and their scopes
        if type(self.agents) in [tuple, list]:
            # single agent
            if len(self.agents) == 1:
                self.num_simultaneous_agents = 1
                self.agents = self.agents[0]
                self.agents_scope = [1]
            else:
                raise ValueError("A list of agents is expected")
        else:
            self.num_simultaneous_agents = 1

    def single_agent_train(self) -> None:
        """Train agent

        - Pre-interaction
        - Compute actions
        - Interact with the environments
        - Render scene
        - Record transitions
        - Post-interaction
        - Reset environments
        """
        assert self.num_simultaneous_agents == 1, "This method is not allowed for simultaneous agents"
        assert self.env.num_agents == 1, "This method is not allowed for multi-agents"

        # reset env
        states, infos = self.env.reset()

        ep_reward = 0.0
        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):

            with contextlib.nullcontext():
                # compute actions
                actions = self.agents.act(states, timestep=timestep, timesteps=self.timesteps)[0]

                # TODO: make saving images it configurable
                if self._save_video:
                    self.images.append(self.env.render()[0])

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)
                ep_reward += rewards[0]

            # record the environments' transitions
            self.agents.record_transition(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
                infos=infos,
                timestep=timestep,
                timesteps=self.timesteps,
            )

            if terminated[0] or truncated[0]:
                if self._save_video:
                    output_filename = f"{self.output_folder}/output_{timestep}_reward_{ep_reward}.mp4"
                    self.save_video(self.images, filename=output_filename, fps=30)
                    self.images.clear()
                    ep_reward = 0.0

            # post-interaction
            self.agents.post_interaction(
                timestep=timestep, timesteps=self.timesteps)

            # reset environments
            if self.env.num_envs > 1:
                states = next_states
            else:
                if terminated.any() or truncated.any():
                    with contextlib.nullcontext():
                        states, infos = self.env.reset()
                else:
                    states = next_states

    def save_video(self, image_list, filename="output.mp4", fps=30):
        if not image_list:
            raise ValueError("image_list is empty")

        # Get frame size from first image
        height, width, _ = image_list[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # for .mp4
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

        for img in image_list:
            # Convert from RGB (matplotlib format) to BGR (OpenCV format)
            img = np.array(img)
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            out.write(bgr.astype(np.uint8))

        out.release()
        output_path = os.path.abspath(filename)
        print(f"Saved video to {output_path}")
