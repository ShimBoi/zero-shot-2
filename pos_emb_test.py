import jax
import jax.numpy as jnp
import argparse
import os
import cv2
import simpler_env
import numpy as np
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from improve.custom.models.rt1.rtx_wrapper import RTXPPO

parser = argparse.ArgumentParser()
parser.add_argument("--exp", type=str, default="default", help="Experiment name")
args = parser.parse_args()

# Setup output directory
env = simpler_env.make('google_robot_pick_coke_can')
device = jax.devices()[0]
print(f"Using device: {device}")

instruction = env.get_language_instruction()

for seqlen in range(1, 16, 1):
    if seqlen != 8:
        continue

    print(f"Testing with sequence length: {seqlen}")
    save_dir = os.path.join("videos", f"seqlen_{seqlen}_{args.exp}")
    os.makedirs(save_dir, exist_ok=True)

    model = RTXPPO(
        env.observation_space["image"]["overhead_camera"]["rgb"], 
        env.action_space, 
        device, 
        task=instruction,
        clip_actions=False,
        clip_values=False,
        reduction="sum",
        seqlen=seqlen
    )
    model.init_state_dict("a2c")

    # Collect up to 10 rollouts
    for rollout_idx in range(10):
        obs, reset_info = env.reset()
        frames = []  # store images for video

        done, truncated = False, False
        while not (done or truncated):
            image = get_image_from_maniskill2_obs_dict(env, obs)
            frames.append(image[..., ::-1])  # convert RGB -> BGR for OpenCV

            image = image / 255.0  # normalize to [0, 1]
            image = jnp.array(image, dtype=jnp.float32)
            image = jnp.expand_dims(image, 0)  # add batch dimension    

            action, _, out = model.act({"states": image}, role="rollout")
            action = np.array(action.block_until_ready(), dtype=np.float64)
            action = action.reshape(env.action_space.shape)

            obs, reward, done, truncated, info = env.step(action)
            model.rng = out["rng"]

        # Save rollout as video
        if frames:
            h, w, _ = frames[0].shape
            out = "success" if info["success"] else "fail"
            out_path = os.path.join(save_dir, f"rollout_{rollout_idx}_{out}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, 30, (w, h))
            for frame in frames:
                writer.write(np.uint8(frame))
            writer.release()
            print(f"Saved rollout video: {out_path}")

env.close()