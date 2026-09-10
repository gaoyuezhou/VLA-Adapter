import gym
import torch
import numpy as np
from gym import spaces


class BlockPushStateWrapper(gym.Wrapper):
    def __init__(self, env, *args, state_normalizer=None, **kwargs):
        super().__init__(env)
        self.env = env
        self.max_steps = 300
        self.step_idx = 0
        # 16-dim state space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32
        )
        # Verify the underlying env is multimodal to ensure keys exist
        assert hasattr(env, "observation_space"), "Env must have observation_space"
        self.state_normalizer = state_normalizer

    def reset(self, goal_idx=None, *args, **kwargs):
        obs = self.env.reset()
        self.step_idx = 0
        return self._flatten_and_normalize(obs)

    def step(self, action):
        # Scale actions to match dataset convention:
        # dataset divides raw actions by 0.03, so model predicts in that scale.
        # Multiply back for the env.
        action = action * 0.03
        obs, reward, done, info = self.env.step(action)

        self.step_idx += 1
        if self.step_idx >= self.max_steps:
            done = True

        info["moved"] = self.env.moved
        info["entered"] = self.env.entered
        info["all_completions_ids"] = []
        info["image"] = self.env.render(mode="rgb_array")

        return self._flatten_and_normalize(obs), reward, done, info

    def _flatten_and_normalize(self, obs):
        """Flatten OrderedDict obs to 16-dim vector and normalize.

        Order must match datasets.block_pushing_state.BlockPushStateDataset.
        """
        keys = [
            "block_translation",          # 2
            "block_orientation",          # 1
            "block2_translation",         # 2
            "block2_orientation",         # 1
            "effector_translation",       # 2
            "effector_target_translation",# 2
            "target_translation",         # 2
            "target_orientation",         # 1
            "target2_translation",        # 2
            "target2_orientation",        # 1
        ]

        flat_obs = []
        for k in keys:
            assert k in obs, f"Observation key {k} missing from env output"
            val = np.asarray(obs[k], dtype=np.float32).flatten()
            flat_obs.append(val)

        flat_obs = np.concatenate(flat_obs).astype(np.float32)

        if self.state_normalizer is not None:
            t = torch.from_numpy(flat_obs).float().unsqueeze(0)
            t = self.state_normalizer.normalize(t)
            flat_obs = t.squeeze(0).numpy()

        return flat_obs

    def seed(self, seed=None):
        return self.env.seed(seed)
