import numpy as np
import gym
from utils import aggregate_dct

class DummyEnv(gym.Env):
    """A minimal environment used for testing."""
    def __init__(
        self,
        action_dim=7,
        state_dim=20,
        proprio_dim=7,
        visual_obs_shape=(224, 224, 3),
        state_based=True,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.proprio_dim = proprio_dim
        if state_based:
            self.observation_shape = (state_dim,)
        else:
            self.observation_shape = visual_obs_shape
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(
            {
                "visual": gym.spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=self.observation_shape,
                    dtype=np.float32,
                ),
                "proprio": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.proprio_dim,),
                    dtype=np.float32,
                ),
            }
        )

    def sample_random_init_goal_states(self, seed):
        rng = np.random.RandomState(seed)
        init_state = rng.randn(self.state_dim).astype(np.float32)
        goal_state = rng.randn(self.state_dim).astype(np.float32)
        return init_state, goal_state

    def update_env(self, env_info):
        pass

    def is_success(self, goal_state, cur_state):
        return True

    def _get_obs(self):
        visual = np.zeros(self.observation_shape, dtype=np.float32)
        proprio = np.zeros(self.proprio_dim, dtype=np.float32)
        state = np.zeros(self.state_dim, dtype=np.float32)
        return {"visual": visual, "proprio": proprio, "state": state}

    def reset(self):
        obs = self._get_obs()
        return obs

    def step(self, action):
        obs = self._get_obs()
        reward = 0.0
        done = False
        info = {"state": np.zeros(self.state_dim, dtype=np.float32)}
        return obs, reward, done, info

    # The following helpers match the interface expected by SerialVectorEnv
    def prepare(self, seed, init_state):
        self.seed(seed)
        obs = self.reset()
        state = np.zeros(self.state_dim, dtype=np.float32)
        return obs, state

    def step_multiple(self, actions):
        obses = []
        rewards = []
        dones = []
        infos = []
        for act in actions:
            o, r, d, info = self.step(act)
            obses.append(o)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
        obses = aggregate_dct(obses)
        rewards = np.array(rewards)
        dones = np.array(dones)
        infos = aggregate_dct(infos)
        return obses, rewards, dones, infos

    def rollout(self, seed, init_state, actions):
        obs, state = self.prepare(seed, init_state)
        obses, rewards, dones, infos = self.step_multiple(actions)
        for k in obses.keys():
            obses[k] = np.vstack([np.expand_dims(obs[k], 0), obses[k]])
        states = np.vstack([np.expand_dims(state, 0), infos["state"]])
        return obses, states
