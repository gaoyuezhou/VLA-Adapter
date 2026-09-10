import gym
import torch
import numpy as np
import envs.cube.cube_env  # noqa: F401 — triggers gymnasium registration


class CubeStateWrapper(gym.Wrapper):
    def __init__(self, env, id=None, task_id=None, max_steps=300, movement_threshold=1e-3,
                 state_normalizer=None):
        super(CubeStateWrapper, self).__init__(env)
        self.env = env
        self.task_id = task_id
        self.max_steps = int(max_steps)
        self.step_idx = 0

        self.moved = 0
        self.entered = 0
        self._movement_threshold = float(movement_threshold)

        self._init_positions = None
        self._moved_flags = None
        self._entered_flags = None

        self.state_normalizer = state_normalizer

    def _normalize_state(self, state):
        if self.state_normalizer is not None:
            t = torch.from_numpy(state).float().unsqueeze(0)
            t = self.state_normalizer.normalize(t)
            return t.squeeze(0).numpy()
        return state

    def _extract_state(self, obs):
        if isinstance(obs, dict):
            state = obs.get("latent", obs)
        else:
            state = obs
        return np.asarray(state, dtype=np.float32)

    def reset(self, goal_idx=None, *args, **kwargs):
        out = self.env.reset(options={'task_id': self.task_id}, **kwargs)

        if isinstance(out, tuple) and len(out) == 2:
            obs, _ = out
        else:
            obs = out

        self.step_idx = 0

        state = self._extract_state(obs)

        # snapshot initial cube positions directly from the mujoco data
        n = int(self.env.unwrapped._num_cubes)
        init_pos = []
        for i in range(n):
            init_pos.append(self.env.unwrapped._data.joint(f"object_joint_{i}").qpos[:3].copy())
        self._init_positions = np.asarray(init_pos)

        # reset flags and counters
        self._moved_flags = np.zeros(n, dtype=bool)
        self._entered_flags = np.zeros(n, dtype=bool)
        self.moved = 0
        self.entered = 0

        return self._normalize_state(state)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        state = self._extract_state(obs)
        state = self._normalize_state(state)

        info["image"] = self.env.render()
        info["all_completions_ids"] = []

        done = False
        if truncated or terminated:
            done = True

        self.step_idx += 1
        if self.step_idx >= self.max_steps:
            done = True

        # moved detection (first time per cube)
        n = int(self.env.unwrapped._num_cubes)
        cur_pos = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            cur_pos[i] = self.env.unwrapped._data.joint(f"object_joint_{i}").qpos[:3].copy()

        deltas = np.linalg.norm(cur_pos[:, :2] - self._init_positions[:, :2], axis=1)
        newly_moved = (~self._moved_flags) & (deltas > self._movement_threshold)
        if newly_moved.any():
            self._moved_flags[newly_moved] = True
            count_new = int(newly_moved.sum())
            self.moved += count_new

        # entered detection
        successes = np.array(self.env.unwrapped._compute_successes(), dtype=bool)
        newly_entered = (~self._entered_flags) & successes
        if newly_entered.any():
            self._entered_flags[newly_entered] = True
            count_new = int(newly_entered.sum())
            self.entered += count_new

        info["moved"] = int(self.moved)
        info["entered"] = int(self.entered)
        info["all_completions_ids"] = []

        return state, reward, done, info

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)
