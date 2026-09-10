import gym
import numpy as np

class RealRobotDummyEnv(gym.Env):
    def __init__(self, obs_dim=20, act_dim=7, goal_dim=0, views=1, id="dummy"):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(views, 3, 224, 224)) # Image N V C H W
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(act_dim,))
        self.views = views

    def reset(self, goal_idx=None):
        # Return random image (V, C, H, W)
        return np.random.rand(self.views, 3, 224, 224).astype(np.float32)

    def step(self, action):
        obs = np.random.rand(self.views, 3, 224, 224).astype(np.float32)
        reward = 0.0
        done = True
        info = {
            "image": np.zeros((224, 224, 3), dtype=np.uint8), # For video recorder
            "max_coverage": 0.0,
            "final_coverage": 0.0,
            "entered": 0.0,
            "moved": 0.0,
            "all_completions_ids": []
        }
        return obs, reward, done, info

    def seed(self, seed=None):
        pass