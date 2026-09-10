import os
import gym
import einops
import torch
import numpy as np
from . import benchmark
from . import get_libero_path
from .envs.env_wrapper import OffScreenRenderEnv
from datasets.libero_state import MAX_STATE_DIM

GOAL_PREDICATES = {
    "open_the_middle_drawer_of_the_cabinet": [
        ["open", "wooden_cabinet_1_middle_region"]
    ],
    "open_the_top_drawer_and_put_the_bowl_inside": [
        ["in", "akita_black_bowl_1", "wooden_cabinet_1_top_region"]
    ],
    "push_the_plate_to_the_front_of_the_stove": [
        ["on", "plate_1", "main_table_stove_front_region"]
    ],
    "put_the_bowl_on_the_plate": [["on", "akita_black_bowl_1", "plate_1"]],
    "put_the_bowl_on_the_stove": [
        ["on", "akita_black_bowl_1", "flat_stove_1_cook_region"]
    ],
    "put_the_bowl_on_top_of_the_cabinet": [
        ["on", "akita_black_bowl_1", "wooden_cabinet_1_top_side"]
    ],
    "put_the_cream_cheese_in_the_bowl": [
        ["on", "cream_cheese_1", "akita_black_bowl_1"]
    ],
    "put_the_wine_bottle_on_the_rack": [
        ["on", "wine_bottle_1", "wine_rack_1_top_region"]
    ],
    "put_the_wine_bottle_on_top_of_the_cabinet": [
        ["on", "wine_bottle_1", "wooden_cabinet_1_top_side"]
    ],
    "turn_on_the_stove": [["turnon", "flat_stove_1"]],
}
IMAGE_SIZE = 224


class LiberoStateEnv(gym.Env):
    """
    A wrapper for OffScreenRenderEnv to initialize environment based on task suite and task name.
    Returns normalized 128-dim state vectors instead of images.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self, task_suite_name="libero_goal", image_size=IMAGE_SIZE, id="libero_goal", view_idx=None,
        state_normalizer=None,
    ):
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        self.image_size = image_size
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(MAX_STATE_DIM,), dtype=np.float32
        )
        self.task_names = list(GOAL_PREDICATES.keys())
        self.benchmark_dict = benchmark.get_benchmark_dict()
        self.task_suite = self.benchmark_dict[task_suite_name]()
        self.env = None
        self.goal_predicates = GOAL_PREDICATES
        self.steps = 0
        self.goal_idx = 0
        self.episodes = 0
        self.state_normalizer = state_normalizer

    def seed(self, seed=None):
        self._seed = seed

        # reset the episode count every time we seed
        # this is done in the main loop for every eval_on_env
        self.episodes = 0

    def _extract_state(self, obs):
        """Extract and normalize 128-dim state from raw robosuite observation dict."""
        joint_pos = np.asarray(obs["robot0_joint_pos"], dtype=np.float32)       # 7
        eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)           # 3
        eef_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32)         # 4
        gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32) # 2
        object_state = np.asarray(obs["object-state"], dtype=np.float32)        # variable

        # Concatenate to match dataset ordering
        state = np.concatenate([joint_pos, eef_pos, eef_quat, gripper_qpos, object_state])

        # Pad to MAX_STATE_DIM
        if len(state) < MAX_STATE_DIM:
            state = np.concatenate([state, np.zeros(MAX_STATE_DIM - len(state), dtype=np.float32)])

        # Normalize using data-driven normalizer
        if self.state_normalizer is not None:
            t = torch.from_numpy(state).float().unsqueeze(0)
            t = self.state_normalizer.normalize(t)
            return t.squeeze(0).numpy()
        return state

    def reset(self, goal_idx, seed=None):
        self.episodes += 1
        self.goal_idx = goal_idx
        self.steps = 0
        task_name = self.task_names[goal_idx]
        task_bddl_file = self._get_task_bddl_file(task_name)

        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": self.image_size,
            "camera_widths": self.image_size,
        }

        self.env = OffScreenRenderEnv(**env_args)
        self.env.seed(self._seed + self.episodes)
        obs = self.env.reset()
        zero_action = np.zeros(7)
        for i in range(20):
            obs, _, _, _ = self.env.step(zero_action)  # make sure objects are stable
        self.finished_tasks = {task_name: False for task_name in self.task_names}
        return self._extract_state(obs)

    def step(self, action):
        self.steps += 1
        obs, _, done, info = self.env.step(action)
        done = done or self.steps >= 300
        info["state"] = obs
        state = self._extract_state(obs)
        reward, info["task_rewards"] = self.get_rewards()
        info["finished_tasks"] = self.finished_tasks.copy()
        info["image"] = einops.rearrange(self._get_img_obs(obs), "V C H W -> H (V W) C")
        info["all_completions_ids"] = []

        cur_task = self.task_names[self.goal_idx]
        info["all_completions_ids"] = self.finished_tasks[cur_task]
        return state, reward, done, info

    def close(self):
        self.env.close()
        self.env = None

    def render(self, mode="rgb_array"):
        obs = self.env.env._get_observations()
        obs = self._get_img_obs(obs, channel_first=False)
        return np.concatenate((obs[0], obs[1]), axis=1).astype(np.uint8)

    def _get_img_obs(self, obs, flip=True, channel_first=True):
        if flip:
            obs["agentview_image"] = obs["agentview_image"][::-1]
            obs["robot0_eye_in_hand_image"] = obs["robot0_eye_in_hand_image"][::-1]
        obs = np.stack(
            [obs["agentview_image"], obs["robot0_eye_in_hand_image"]], axis=0
        )
        if channel_first:
            obs = einops.rearrange(obs, "V H W C -> V C H W")
        return obs

    def _get_task_bddl_file(self, task_name):
        task_id = self.task_suite.get_task_names().index(task_name)
        task = self.task_suite.get_task(task_id)
        task_bddl_file = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )
        return task_bddl_file

    def get_rewards(self):
        task_rewards = {}
        for task, goal_states in self.goal_predicates.items():
            task_completed = self.env.env._eval_predicate(goal_states[0])
            task_rewards[task] = int(task_completed and not self.finished_tasks[task])
            self.finished_tasks[task] = self.finished_tasks[task] or task_completed

        cur_task = self.task_names[self.goal_idx]
        reward = task_rewards[cur_task]
        task_rewards = {cur_task: task_rewards[cur_task]}
        return reward, task_rewards
