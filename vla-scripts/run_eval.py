"""
run_eval.py

Evaluates a fine-tuned VLA-Adapter policy in the patch_policy simulation environments
(BlockPush, Cube, LIBERO Goal, PushT). Ported from the openvla-oft fork.
"""

import logging
import os
import sys

# Environment defaults for the vendored sim envs (must be set before `envs.*` is imported):
#   ASSET_PATH          -> pybullet URDF/OBJ assets for BlockPush
#   LIBERO_CONFIG_PATH  -> where the vendored LIBERO writes its config.yaml (keeps paths inside this repo instead of
#                          colliding with a pip-installed / other LIBERO copy in ~/.libero)
#   MUJOCO_GL           -> headless rendering for Cube / LIBERO
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ASSET_PATH", os.path.join(_REPO_ROOT, "envs", "assets"))
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.join(_REPO_ROOT, "envs", "libero", ".libero_config"))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import imageio

import einops
import hydra
import numpy as np
import torch
import tqdm
from omegaconf import DictConfig, OmegaConf

import wandb

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
import prismatic.vla.constants as C
from prismatic.vla.constants import set_constants


# Task descriptions for single-task environments
TASK_DESCRIPTIONS = {
    "blockpush": "push each block to its target location",
    "cube": "stack the cubes to match their target locations",
    "pusht": "push the T-shaped block to the target",
    "franka_insertion": "insert the object into the target",
}


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def validate_config(cfg: DictConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None and cfg.pretrained_checkpoint != "", (
        "pretrained_checkpoint must be specified!"
    )

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    assert cfg.env_name in ["blockpush", "cube", "libero_goal", "pusht"], (
        f"Invalid env_name: {cfg.env_name}. Must be one of: blockpush, cube, libero_goal, pusht"
    )


def initialize_model(cfg: DictConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)

    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=cfg.proprio_dim,
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor


def check_unnorm_key(cfg: DictConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Initialize unnorm_key
    unnorm_key = cfg.unnorm_key

    # If not specified, try to infer from env_name
    if not unnorm_key:
        # Use env_name as the unnorm key (matches dataset_name used during training)
        unnorm_key = cfg.env_name

    # Check if key exists in model norm_stats, try common variants
    if unnorm_key not in model.norm_stats:
        # Try without underscores
        candidates = [unnorm_key, unnorm_key.replace("_", "")]
        # Try all keys if only one exists
        if len(model.norm_stats) == 1:
            unnorm_key = next(iter(model.norm_stats.keys()))
        else:
            found = False
            for candidate in candidates:
                if candidate in model.norm_stats:
                    unnorm_key = candidate
                    found = True
                    break
            if not found:
                raise ValueError(
                    f"Action un-norm key '{unnorm_key}' not found in VLA norm_stats. "
                    f"Available keys: {list(model.norm_stats.keys())}. "
                    f"Set unnorm_key= to one of these."
                )

    # Set the unnorm_key in cfg
    OmegaConf.set_struct(cfg, False)
    cfg.unnorm_key = unnorm_key


def setup_logging(cfg: DictConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-{cfg.env_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging if enabled
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
            settings=wandb.Settings(init_timeout=300),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def save_video(frames: List[np.ndarray], video_path: str, fps: int = 10) -> None:
    """Save a list of frames as a video file.

    Args:
        frames: List of RGB images as numpy arrays (H, W, C) with uint8 dtype.
        video_path: Path to save the video file.
        fps: Frames per second for the video.
    """
    if not frames:
        logger.warning(f"No frames to save for video: {video_path}")
        return

    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    imageio.mimwrite(video_path, frames, fps=fps, format="FFMPEG")
    logger.info(f"Saved video to: {video_path}")


def create_env(cfg: DictConfig):
    """Create and return the evaluation environment."""
    if cfg.env_name == "blockpush":
        from envs.block_pushing.block_pushing_multimodal import BlockPushMultimodalMultiview

        env = BlockPushMultimodalMultiview(id="blockpush")
        return env

    elif cfg.env_name == "cube":
        import gymnasium
        from envs.cube.cube_env import CubeWrapper

        inner_env = gymnasium.make(
            f"cube-{cfg.cube_env_type}-v1",
            reward_task_id=cfg.cube_task_id,
            max_episode_steps=cfg.max_steps,
            mode="task",
        )
        env = CubeWrapper(inner_env, id="cube", task_id=cfg.cube_task_id)
        return env

    elif cfg.env_name == "pusht":
        from envs.pusht import PushTKeypointsEnv, PushWrapper

        inner_env = PushTKeypointsEnv()
        env = PushWrapper(inner_env, id="pusht")
        return env

    elif cfg.env_name == "libero_goal":
        from envs.libero.libero_env import LiberoEnv

        env = LiberoEnv(task_suite_name="libero_goal")
        return env

    elif cfg.env_name == "franka_insertion":
        # Dummy stand-in; real eval happens off-script on the actual Franka.
        from envs.env_utils import RealRobotDummyEnv

        env = RealRobotDummyEnv(act_dim=cfg.action_dim, views=cfg.num_images_in_input, id="franka_insertion")
        return env

    else:
        raise ValueError(f"Unknown environment: {cfg.env_name}")


def prepare_observation(obs, cfg: DictConfig, resize_size, info=None):
    """Prepare observation for policy input.

    Converts (V, C, H, W) float [0,1] tensor from env wrappers
    to the dict format expected by get_vla_action.
    """
    # obs is (V, C, H, W) float [0,1] from env wrapper
    # Convert primary view to uint8 HWC for get_vla_action
    primary = obs[0]  # (C, H, W)
    img = einops.rearrange(primary, "C H W -> H W C")
    img = (img * 255).astype(np.uint8)

    # Resize image to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)

    # Prepare observations dict
    observation = {"full_image": img_resized}

    # Add wrist image if multi-view
    if obs.shape[0] > 1 and cfg.num_images_in_input > 1:
        wrist = obs[1]  # (C, H, W)
        wrist_img = einops.rearrange(wrist, "C H W -> H W C")
        wrist_img = (wrist_img * 255).astype(np.uint8)
        observation["wrist_image"] = resize_image_for_policy(wrist_img, resize_size)

    # Add proprioception state if used (LIBERO)
    if cfg.use_proprio and info is not None and "state" in info:
        from experiments.robot.libero.libero_utils import quat2axisangle

        raw_obs = info["state"]
        observation["state"] = np.concatenate(
            (raw_obs["robot0_eef_pos"], quat2axisangle(raw_obs["robot0_eef_quat"]), raw_obs["robot0_gripper_qpos"])
        )

    return observation, img  # Return both processed observation and original image for replay


def process_action(action, cfg: DictConfig):
    """Process action before sending to environment."""
    # LIBERO: normalize gripper action [0,1] -> [-1,+1] and invert sign
    # (same as run_libero_eval.process_action)
    if cfg.env_name == "libero_goal":
        action = normalize_gripper_action(action, binarize=True)
        # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
        # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
        action = invert_gripper_action(action)

    return action


def check_success(cfg: DictConfig, done: bool, info: Dict[str, Any], env=None, task_description: str = "") -> bool:
    """Check if the episode was successful (env-specific)."""
    if cfg.env_name == "libero_goal":
        # Use finished_tasks (persistent) rather than task_rewards (one-shot).
        # task_rewards only gives reward=1 on the first step the predicate becomes True,
        # so checking it at episode end would miss earlier completions.
        finished_tasks = info.get("finished_tasks", {})
        task_key = task_description.replace(" ", "_")
        return bool(finished_tasks.get(task_key, False))
    elif cfg.env_name == "blockpush":
        return info.get("entered", 0) >= 2  # Both blocks in targets
    elif cfg.env_name == "cube":
        # Success requires all cubes to have entered their targets
        num_cubes = getattr(env.unwrapped, "_num_cubes", 1) if env is not None else 1
        return info.get("entered", 0) >= num_cubes
    elif cfg.env_name == "pusht":
        return info.get("max_coverage", 0) >= 0.8  # 80% coverage threshold
    return False


def get_task_descriptions(cfg: DictConfig) -> List[Tuple[int, str]]:
    """Get list of (task_idx, task_description) for evaluation."""
    if cfg.env_name == "libero_goal":
        # 10 LIBERO tasks
        from envs.libero.libero_env import GOAL_PREDICATES

        task_names = list(GOAL_PREDICATES.keys())
        return [(i, name.replace("_", " ")) for i, name in enumerate(task_names)]
    else:
        # Single-task environments
        description = TASK_DESCRIPTIONS[cfg.env_name]
        return [(0, description)]


def run_episode(
    cfg: DictConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    goal_idx: int = 0,
    log_file=None,
):
    """Run a single episode in the environment."""
    print(goal_idx)
    # Reset environment (BlockPush doesn't accept goal_idx)
    # sample random seed
    env.seed(np.random.randint(0, 10000))
    if cfg.env_name == "blockpush":
        obs = env.reset()
    else:
        obs = env.reset(goal_idx=goal_idx)

    # Initialize action queue
    if cfg.num_open_loop_steps != C.NUM_ACTIONS_CHUNK:
        print(
            f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the C.NUM_ACTIONS_CHUNK "
            f"({C.NUM_ACTIONS_CHUNK}) constant defined in prismatic.vla.constants! For best performance (in terms of "
            "both speed and success rate), we recommend executing the full action chunk."
        )
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    t = 0
    done = False
    replay_images = []
    last_info = {}
    total_reward = 0

    # Run episode
    success = False
    try:
      while t < cfg.max_steps + cfg.num_steps_wait:
          # Do nothing for the first few timesteps to let objects stabilize
          if t < cfg.num_steps_wait:
              dummy_action = np.zeros(cfg.action_dim)
              if cfg.env_name == "libero_goal":
                  dummy_action[-1] = -1  # gripper open
              obs, reward, done, info = env.step(dummy_action.tolist())
              total_reward += reward
              last_info = info
              t += 1
              continue

          # Prepare observation (pass info for proprio extraction if needed)
          observation, img = prepare_observation(obs, cfg, resize_size, info=last_info)
          replay_images.append(img)

          # If action queue is empty, requery model
          if len(action_queue) == 0:
              # Query model to get action
              actions = get_action(
                  cfg,
                  model,
                  observation,
                  task_description,
                  processor=processor,
                  action_head=action_head,
                  proprio_projector=proprio_projector,
                  noisy_action_projector=noisy_action_projector,
                  use_film=cfg.use_film,
                  use_minivlm=cfg.get("use_minivlm", True),
              )
              action_queue.extend(actions)

          # Get action from queue
          action = action_queue.popleft()

          # Process action
          action = process_action(action, cfg)

          # Execute action in environment
          # obs, reward, done, info = env.step(action.tolist())
          obs, reward, done, info = env.step(action)
          total_reward += reward
          last_info = info
          if done:
              success = check_success(cfg, done, info, env=env, task_description=task_description)
              break
          t += 1

      # Final success check for envs that don't set done=True on success
      if not success:
        success = check_success(cfg, done, last_info, env=env, task_description=task_description)

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)
        raise

    # Collect per-episode metrics (matching patch_policy's online_eval.py)
    episode_metrics = {"reward": total_reward}
    if cfg.env_name in ("blockpush", "cube"):
        episode_metrics["moved"] = last_info.get("moved", 0)
        episode_metrics["entered"] = last_info.get("entered", 0)
    elif cfg.env_name == "pusht":
        episode_metrics["max_coverage"] = last_info.get("max_coverage", 0)
        episode_metrics["final_coverage"] = last_info.get("final_coverage", 0)

    return success, replay_images, episode_metrics


def run_task(
    cfg: DictConfig,
    env,
    task_idx: int,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    total_episodes: int = 0,
    total_successes: int = 0,
    log_file=None,
    video_dir: Optional[str] = None,
):
    """Run evaluation for a single task.

    Args:
        video_dir: If provided, save episode videos to this directory.
    """
    # Start episodes
    task_episodes, task_successes = 0, 0
    all_episode_metrics = []
    for episode_idx in tqdm.tqdm(range(cfg.num_trials)):
        log_message(f"\nTask: {task_description}", log_file)
        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        # Run episode
        success, replay_images, episode_metrics = run_episode(
            cfg,
            env,
            task_description,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            goal_idx=task_idx,
            log_file=log_file,
        )

        # Save video with success/failure status in filename
        if video_dir and replay_images:
            task_name_clean = task_description.replace(" ", "_").replace("/", "_")
            status = "success" if success else "failure"
            video_path = os.path.join(
                video_dir,
                f"task{task_idx:02d}_{task_name_clean}",
                f"episode_{episode_idx:03d}_{status}.mp4",
            )
            save_video(replay_images, video_path, fps=cfg.get("video_fps", 10))
            log_message(f"Video saved: {video_path}", log_file)

        # Update counters
        task_episodes += 1
        total_episodes += 1
        all_episode_metrics.append(episode_metrics)
        if success:
            task_successes += 1
            total_successes += 1

        # Log results
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    # Log task results
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    # Log to wandb if enabled (matching patch_policy's online_eval.py metric format)
    if cfg.use_wandb and all_episode_metrics:
        # Average reward (matches patch_policy's "eval_on_env")
        rewards = [m["reward"] for m in all_episode_metrics]
        avg_reward = sum(rewards) / len(rewards)
        wandb_metrics = {
            f"eval_on_env": avg_reward,
            f"success_rate": task_success_rate,
            f"num_episodes": task_episodes,
        }

        # Env-specific metrics (matches patch_policy lines 345-359)
        if cfg.env_name in ("blockpush", "cube"):
            metric_final = "entered"
            metric_max = "moved"
        elif cfg.env_name == "pusht":
            metric_final = "final coverage"
            metric_max = "max coverage"
        else:
            metric_final = None

        if metric_final is not None:
            final_values = [
                m.get("entered" if metric_final == "entered" else "final_coverage", 0) for m in all_episode_metrics
            ]
            max_values = [m.get("moved" if metric_max == "moved" else "max_coverage", 0) for m in all_episode_metrics]
            wandb_metrics.update(
                {
                    f"{metric_final} mean": sum(final_values) / len(final_values),
                    f"{metric_final} max": max(final_values),
                    f"{metric_final} min": min(final_values),
                    f"{metric_max} mean": sum(max_values) / len(max_values),
                    f"{metric_max} max": max(max_values),
                    f"{metric_max} min": min(max_values),
                }
            )

        wandb.log(wandb_metrics)
        print("####### FINAL EVAL")
        print(wandb_metrics)

    return total_episodes, total_successes, all_episode_metrics


@hydra.main(version_base="1.2", config_path="configs", config_name="eval")
def run_eval(cfg: DictConfig) -> float:
    """Main function to evaluate a fine-tuned policy in simulation environments."""
    # Set robot constants from config
    set_constants(
        action_dim=cfg.action_dim,
        num_actions_chunk=cfg.num_actions_chunk,
        proprio_dim=cfg.proprio_dim,
        normalization_type=cfg.normalization_type,
    )

    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Initialize model and components
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    # Create environment
    env = create_env(cfg)

    # Get tasks to evaluate
    tasks = get_task_descriptions(cfg)

    log_message(f"Environment: {cfg.env_name}", log_file)
    log_message(f"Number of tasks: {len(tasks)}", log_file)

    # Prepare video directory if saving videos
    video_dir = None
    if cfg.get("save_videos", True):
        video_dir = os.path.join(cfg.local_log_dir, run_id, "videos")
        os.makedirs(video_dir, exist_ok=True)
        log_message(f"Saving evaluation videos to: {video_dir}", log_file)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    all_metrics = []
    for task_idx, task_description in tqdm.tqdm(tasks):
        total_episodes, total_successes, task_metrics = run_task(
            cfg,
            env,
            task_idx,
            task_description,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            total_episodes,
            total_successes,
            log_file,
            video_dir=video_dir,
        )
        all_metrics.extend(task_metrics)

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    # Log to wandb if enabled (matching patch_policy's online_eval.py metric format)
    if cfg.use_wandb and all_metrics:
        rewards = [m["reward"] for m in all_metrics]
        avg_reward = sum(rewards) / len(rewards)
        final_wandb = {
            "eval_on_env/total": avg_reward,
            "success_rate/total": final_success_rate,
            "num_episodes/total": total_episodes,
        }

        if cfg.env_name in ("blockpush", "cube"):
            metric_final, metric_max = "entered", "moved"
        elif cfg.env_name == "pusht":
            metric_final, metric_max = "final coverage", "max coverage"
        else:
            metric_final = None

        if metric_final is not None:
            final_values = [m.get("entered" if metric_final == "entered" else "final_coverage", 0) for m in all_metrics]
            max_values = [m.get("moved" if metric_max == "moved" else "max_coverage", 0) for m in all_metrics]
            final_wandb.update(
                {
                    f"{metric_final} mean/total": sum(final_values) / len(final_values),
                    f"{metric_final} max/total": max(final_values),
                    f"{metric_final} min/total": min(final_values),
                    f"{metric_max} mean/total": sum(max_values) / len(max_values),
                    f"{metric_max} max/total": max(max_values),
                    f"{metric_max} min/total": min(max_values),
                }
            )

        wandb.log(final_wandb)
        wandb.save(local_log_filepath)

    # Close log file
    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    run_eval()
