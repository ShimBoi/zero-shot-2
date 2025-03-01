import os
import sys
from typing import Optional, Sequence

from contextlib import contextmanager

from skrl import logger
from skrl.envs.loaders.torch.isaacgym_envs import _print_cfg, _omegaconf_to_dict


def make_env(
    task_name: str = "",
    num_envs: Optional[int] = None,
    headless: Optional[bool] = None,
    cli_args: Sequence[str] = [],
    isaacgymenvs_path: str = "",
    show_cfg: bool = True,
    eval: Optional[bool] = False,
):
    """Load an Isaac Gym environment (preview 3)

    Isaac Gym benchmark environments: https://github.com/isaac-sim/IsaacGymEnvs

    :param task_name: The name of the task (default: ``""``).
                      If not specified, the task name is taken from the command line argument (``task=TASK_NAME``).
                      Command line argument has priority over function parameter if both are specified
    :type task_name: str, optional
    :param num_envs: Number of parallel environments to create (default: ``None``).
                     If not specified, the default number of environments defined in the task configuration is used.
                     Command line argument has priority over function parameter if both are specified
    :type num_envs: int, optional
    :param headless: Whether to use headless mode (no rendering) (default: ``None``).
                     If not specified, the default task configuration is used.
                     Command line argument has priority over function parameter if both are specified
    :type headless: bool, optional
    :param cli_args: IsaacGymEnvs configuration and command line arguments (default: ``[]``)
    :type cli_args: list of str, optional
    :param isaacgymenvs_path: The path to the ``isaacgymenvs`` directory (default: ``""``).
                              If empty, the path will obtained from isaacgymenvs package metadata
    :type isaacgymenvs_path: str, optional
    :param show_cfg: Whether to print the configuration (default: ``True``)
    :type show_cfg: bool, optional

    :raises ValueError: The task name has not been defined, neither by the function parameter nor by the command line arguments
    :raises RuntimeError: The isaacgymenvs package is not installed or the path is wrong

    :return: Isaac Gym environment (preview 3)
    :rtype: isaacgymenvs.tasks.base.vec_task.VecTask
    """
    import isaacgym
    import isaacgymenvs
    from hydra._internal.hydra import Hydra
    from hydra._internal.utils import create_automatic_config_search_path, get_args_parser
    from hydra.types import RunMode
    from omegaconf import OmegaConf

    # check task from command line arguments
    defined = False
    for arg in sys.argv:
        if arg.startswith("task="):
            defined = True
            break
    # get task name from command line arguments
    if defined:
        if task_name and task_name != arg.split("task=")[1].split(" ")[0]:
            logger.warning(
                "Overriding task name ({}) with command line argument ({})".format(
                    task_name, arg.split("task=")[1].split(" ")[0]
                )
            )
    # get task name from function arguments
    else:
        if task_name:
            sys.argv.append(f"task={task_name}")
        else:
            raise ValueError(
                "No task name defined. Set task_name parameter or use task=<task_name> as command line argument"
            )

    # check num_envs from command line arguments
    defined = False
    for arg in sys.argv:
        if arg.startswith("num_envs="):
            defined = True
            break
    # get num_envs from command line arguments
    if defined:
        if num_envs is not None and num_envs != int(arg.split("num_envs=")[1].split(" ")[0]):
            logger.warning(
                "Overriding num_envs ({}) with command line argument (num_envs={})".format(
                    num_envs, arg.split("num_envs=")[1].split(" ")[0]
                )
            )
    # get num_envs from function arguments
    elif num_envs is not None and num_envs > 0:
        sys.argv.append(f"num_envs={num_envs}")

    # check headless from command line arguments
    defined = False
    for arg in sys.argv:
        if arg.startswith("headless="):
            defined = True
            break
    # get headless from command line arguments
    if defined:
        if headless is not None and str(headless).lower() != arg.split("headless=")[1].split(" ")[0].lower():
            logger.warning(
                "Overriding headless ({}) with command line argument (headless={})".format(
                    headless, arg.split("headless=")[1].split(" ")[0]
                )
            )
    # get headless from function arguments
    elif headless is not None:
        sys.argv.append(f"headless={headless}")

    # others command line arguments
    sys.argv += cli_args

    # get isaacgymenvs path from isaacgymenvs package metadata
    if isaacgymenvs_path == "":
        if not hasattr(isaacgymenvs, "__path__"):
            raise RuntimeError("isaacgymenvs package is not installed")
        isaacgymenvs_path = list(isaacgymenvs.__path__)[0]
    config_path = os.path.join(isaacgymenvs_path, "cfg")

    # set omegaconf resolvers
    try:
        OmegaConf.register_new_resolver(
            "eq", lambda x, y: x.lower() == y.lower())
    except Exception as e:
        pass
    try:
        OmegaConf.register_new_resolver(
            "contains", lambda x, y: x.lower() in y.lower())
    except Exception as e:
        pass
    try:
        OmegaConf.register_new_resolver(
            "if", lambda condition, a, b: a if condition else b)
    except Exception as e:
        pass
    try:
        OmegaConf.register_new_resolver(
            "resolve_default", lambda default, arg: default if arg == "" else arg)
    except Exception as e:
        pass

    # get hydra config without use @hydra.main
    config_file = "config"
    args = get_args_parser().parse_args()
    search_path = create_automatic_config_search_path(
        config_file, None, config_path)
    hydra_object = Hydra.create_main_hydra2(
        task_name="load_isaacgymenv", config_search_path=search_path)
    config = hydra_object.compose_config(
        config_file, args.overrides, run_mode=RunMode.RUN)

    cfg = _omegaconf_to_dict(config.task)

    # print config
    if show_cfg:
        print(f"\nIsaac Gym environment ({config.task.name})")
        _print_cfg(cfg)

    # load environment
    sys.path.append(isaacgymenvs_path)
    from tasks import isaacgym_task_map  # type: ignore

    try:
        env = isaacgym_task_map[config.task.name](
            cfg=cfg,
            sim_device=config.sim_device,
            graphics_device_id=config.graphics_device_id,
            headless=config.headless,
        )
    except TypeError as e:
        env = isaacgym_task_map[config.task.name](
            cfg=cfg,
            rl_device=config.rl_device,
            sim_device=config.sim_device,
            graphics_device_id=config.graphics_device_id,
            headless=config.headless,
            virtual_screen_capture=config.capture_video,  # TODO: check
            force_render=config.force_render,
        )

    return env
