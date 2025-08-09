import argparse
import os
from pathlib import Path
from tqdm import tqdm
import time

import furniture_bench
from furniture_bench.envs.furniture_sim_env import FurnitureSimEnv
import torch  # needs to be after isaac gym imports
from omegaconf import DictConfig, OmegaConf
from src.behavior.base import Actor  # noqa
from src.behavior.diffusion import DiffusionPolicy  # noqa
from src.eval.rollout import calculate_success_rate
from src.behavior import get_actor
from src.common.tasks import task2idx, task_timeout
from src.common.files import trajectory_save_dir
from src.eval.eval_utils import load_model_weights
# from src.gym import get_rl_env
from typing import Any, List, Optional
from ipdb import set_trace as bp  # noqa
import wandb
from wandb import Api
from wandb.sdk.wandb_run import Run

api = Api(overrides=dict(entity=os.environ.get("WANDB_ENTITY")))

def validate_args(args: argparse.Namespace):
    assert (
        sum(
            [
                args.run_id is not None,
                args.sweep_id is not None,
                args.project_id is not None,
                args.wt_path is not None,
            ]
        )
        == 1
    ), "Exactly one of run-id, sweep-id, project-id must be provided"
    assert args.run_state is None or all(
        [
            state in ["running", "finished", "failed", "crashed"]
            for state in args.run_state
        ]
    ), (
        "Invalid run-state: "
        f"{args.run_state}. Valid options are: None, running, finished, failed, crashed"
    )

    assert not args.leaderboard, "Leaderboard mode is not supported as of now"

    assert not args.store_video_wandb or args.wandb, "store-video-wandb requires wandb"

from pathlib import Path
import furniture_bench  # noqa: F401
from furniture_bench.envs.observation import (
    DEFAULT_VISUAL_OBS,
    DEFAULT_STATE_OBS,
    FULL_OBS,
)
from furniture_bench.envs.furniture_sim_env import FurnitureSimEnv
from furniture_bench.envs.furniture_rl_sim_env import FurnitureRLSimEnv
from src.common.context import suppress_all_output
from ipdb import set_trace as bp

def get_rl_env(
    gpu_id,
    task="one_leg",
    num_envs=1,
    randomness="low",
    max_env_steps=5_000,
    resize_img=True,
    observation_space="image",  # Observation space for the robot. Options are 'image' and 'state'.
    act_rot_repr="quat",
    action_type="pos",  # Action type for the robot. Options are 'delta' and 'pos'.
    april_tags=False,
    verbose=False,
    headless=True,
    record=False,
    **kwargs,
) -> FurnitureRLSimEnv:
    if not april_tags:
        from furniture_bench.envs import furniture_sim_env

        furniture_sim_env.ASSET_ROOT = str(
            Path(__file__).parent.parent.absolute() / "assets"
        )

    # To ensure we can replay the rollouts, we need to (1) include all robot states in the observation space
    # and (2) ensure that the robot state is stored as a dict for compatibility with the teleop data
    obs_keys = FULL_OBS
    if observation_space == "state":
        # Filter out keys with `image` in them
        obs_keys = [key for key in obs_keys if "image" not in key]

    if action_type == "relative":
        print(
            "[INFO] Using relative actions. This keeps the environment using position actions."
        )
    action_type = "pos" if action_type == "relative" else action_type

    env = FurnitureRLSimEnv(
        furniture=task,  # Specifies the type of furniture [lamp | square_table | desk | drawer | cabinet | round_table | stool | chair | one_leg].
        num_envs=num_envs,  # Number of parallel environments.
        resize_img=resize_img,  # If true, images are resized to 224 x 224.
        concat_robot_state=False,  # If true, robot state is concatenated to the observation.
        headless=headless,  # If true, simulation runs without GUI.
        obs_keys=obs_keys,
        compute_device_id=gpu_id,
        graphics_device_id=gpu_id,
        init_assembled=False,  # If true, the environment is initialized with assembled furniture.
        np_step_out=False,  # If true, env.step() returns Numpy arrays.
        channel_first=False,  # If true, images are returned in channel first format.
        randomness=randomness,  # Level of randomness in the environment [low | med | high].
        high_random_idx=-1,  # Index of the high randomness level (range: [0-2]). Default -1 will randomly select the index within the range.
        save_camera_input=False,  # If true, the initial camera inputs are saved.
        record=record,  # If true, videos of the wrist and front cameras' RGB inputs are recorded.
        max_env_steps=max_env_steps,  # Maximum number of steps per episode.
        act_rot_repr=act_rot_repr,  # Representation of rotation for action space. Options are 'quat' and 'axis'.
        ctrl_mode="diffik",  # Control mode for the robot. Options are 'osc' and 'diffik'.
        action_type=action_type,  # Action type for the robot. Options are 'delta' and 'pos'.
        verbose=verbose,  # If true, prints debug information.
        **kwargs,
    )

    return env

from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

from typing import Union, Optional
from dataclasses import dataclass
import tensorflow as tf
import numpy as np
import math

@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on
    
def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def resize_image(img, resize_size):
    """
    Takes numpy array corresponding to a single image and returns resized image as numpy array.

    NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                    the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
    """
    assert isinstance(resize_size, tuple)
    # Resize to image size expected by model
    img = tf.image.encode_jpeg(img)  # Encode as JPEG, as done in RLDS dataset builder
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = img.numpy()
    return img

def get_isaacsim_image(obs, resize_size):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs["color_image2"].cpu().numpy().squeeze(0)
    # img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    img = resize_image(img, resize_size)
    return img

import os
import sys

import imageio
def save_rollout_video(rollout_images, name):
    """Saves an MP4 replay of an episode."""
    os.makedirs("./rollouts", exist_ok=True)
    mp4_path = f"./rollouts/rollout-{name}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=5)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=False, nargs="*")
    parser.add_argument("--wt-path", type=str, default=None)

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--n-rollouts", type=int, default=1)
    parser.add_argument("--randomness", type=str, default="low")
    parser.add_argument(
        "--task",
        "-f",
        type=str,
        choices=[
            "one_leg",
            "lamp",
            "round_table",
            "desk",
            "square_table",
            "cabinet",
            "mug_rack",
            "factory_peg_hole",
        ],
        required=True,
    )
    parser.add_argument("--n-parts-assemble", type=int, default=None)

    parser.add_argument("--save-rollouts", action="store_true")
    parser.add_argument("--save-failures", action="store_true")
    parser.add_argument("--store-full-resolution-video", action="store_true")

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--leaderboard", action="store_true")

    # Define what should be done if the success rate fields are already present
    parser.add_argument(
        "--if-exists",
        type=str,
        choices=["skip", "overwrite", "append", "error"],
        default="error",
    )
    parser.add_argument(
        "--run-state",
        type=str,
        default=None,
        choices=["running", "finished", "failed", "crashed"],
        nargs="*",
    )

    # For batch evaluating runs from a sweep or a project
    parser.add_argument("--sweep-id", type=str, default=None)
    parser.add_argument("--project-id", type=str, default=None)

    parser.add_argument("--continuous-mode", action="store_true")
    parser.add_argument(
        "--continuous-interval",
        type=int,
        default=60,
        help="Pause interval before next evaluation",
    )
    parser.add_argument("--ignore-currently-evaluating-flag", action="store_true")

    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--store-video-wandb", action="store_true")
    parser.add_argument("--eval-top-k", type=int, default=None)
    parser.add_argument(
        "--action-type", type=str, default="pos", choices=["delta", "pos", "relative"]
    )
    parser.add_argument("--prioritize-fewest-rollouts", action="store_true")
    parser.add_argument("--multitask", action="store_true")
    parser.add_argument("--compress-pickles", action="store_true")
    parser.add_argument("--max-rollouts", type=int, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--max-rollout-steps", type=int, default=None)
    parser.add_argument("--april-tags", action="store_true")

    parser.add_argument(
        "--observation-space", choices=["image", "state"], default="state"
    )
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--wt-type", type=str, default="best_success_rate")

    parser.add_argument("--stop-after-n-success", type=int, default=0)
    parser.add_argument("--break-on-n-success", action="store_true")
    parser.add_argument("--record-for-coverage", action="store_true")

    parser.add_argument("--save-rollouts-suffix", type=str, default="")

    # Parse the arguments
    args = parser.parse_args()

    # Validate the arguments
    validate_args(args)

    # Make the device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Set the timeout
    rollout_max_steps = (
        task_timeout(args.task, n_parts=args.n_parts_assemble)
        if args.max_rollout_steps is None
        else args.max_rollout_steps
    )
        
    ### TODO: mult envs causes seg-faults
    env = get_rl_env(
        gpu_id=args.gpu,
        task=args.task,
        num_envs=args.n_envs,
        randomness=args.randomness,
        observation_space=args.observation_space,
        max_env_steps=5_000,
        resize_img=False,
        act_rot_repr="axis",
        action_type=args.action_type,
        april_tags=args.april_tags,
        verbose=args.verbose,
        headless=not args.visualize,
    )
    
    obs = env.reset()
    
    cfg = GenerateConfig()
    cfg.pretrained_checkpoint = args.wt_path
    cfg.unnorm_key = None
    model = get_model(cfg)
    processor = get_processor(cfg)
    
    resize_size = get_image_resize_size(cfg)
    
    from src.common.tasks import simple_task_descriptions
    task_description = simple_task_descriptions[args.task][0]

    images = []
    for _ in tqdm(range(args.n_rollouts)):
        # format obs for openVLA
        img = get_isaacsim_image(obs, resize_size)
        images.append(img)
        robot_state = obs['robot_state']
        observation = {
            "full_image": img,
            "state": np.concatenate(
                (robot_state["ee_pos"].cpu().numpy().squeeze(0), quat2axisangle(robot_state["ee_quat"].cpu().numpy().squeeze(0)), robot_state["gripper_width"].cpu().numpy().squeeze(0))
            ),
        }
        
        action = get_action(
            cfg,
            model,
            observation,
            task_description,
            processor=processor,
        )
        
        action = normalize_gripper_action(action, binarize=True)
        action = invert_gripper_action(action)
        
        action = torch.from_numpy(action).float().to(device)
        action[:6] *= 0.1
        obs, reward, done, info = env.step(action)
        
    env.close()
    print("Done!")
    
    # save video of rollout
    save_rollout_video(images, "test")