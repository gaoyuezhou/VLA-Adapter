"""TrajectoryDataset for bilateral-FACTR teleop episodes (one .pkl per episode).

Each pickle written by factr_ws/launch/collect_data.py has this schema
(see /home_shared/grail_bowen/factr_ws/collect_data.md §4):

    {
      "data": {
        # leader-arm command path
        "/factr_teleop/left/cmd_franka_pos":     [ndarray(7,) float64, ...],   # joint targets,  ~60 Hz
        "/factr_teleop/left/cmd_gripper_pos":    [ndarray(1,) float64, ...],   # leader trigger angle (rad)

        # follower observations (from libfranka via NUC bridge)
        "/franka/left/obs_franka_state":         [ndarray(7,) float64, ...],   # measured joint pos
        "/franka/left/obs_franka_eef_pose":      [ndarray(7,) float64, ...],   # [x,y,z, qx,qy,qz,qw] (O_T_EE, panda_link8)
        "/franka/left/obs_franka_gripper_width": [ndarray(1,) float64, ...],   # measured Franka Hand width (m)
        "/franka/left/obs_franka_torque":        [ndarray(7,) float64, ...],   # external joint torque

        # cameras (JPEG-encoded uint8, ~30 Hz)
        "/realsense/wrist/im": [ndarray(N,) uint8, ...],
        "/realsense/side/im":  [ndarray(N,) uint8, ...],
      },
      "timestamps": {<same keys>: [int ns, ...]},
      ...
    }

The image stream is the canonical clock (T == number of camera frames per
episode). For each image timestamp we nearest-neighbor align one sample from
each non-image topic via bisect on the timestamp arrays.

Output of get_frames (matches the current trajectory dataset contract used in this repo):

    obs:     (T, V, C=3, H, W) float32 in [0, 1]   # V = len(image_topics)
    actions: (T, action_dim)   float32
    mask:    (T,) bool

Wiring into Hydra training:

    dataset:
      dataset_class: factr_pickle.FactrPickleDataset
      data_directory: ${env_vars.datasets.franka_insertion}
      # FactrPickleDataset-specific kwargs (defaults applied if omitted):
      #   image_topics: ["/realsense/wrist/im", "/realsense/side/im"]
      #   action_type: eef_delta     # "joint_absolute" | "joint_delta" | "eef_absolute" | "eef_delta"
      #   rot_unit: axis             # "axis" | "euler"
      #   frame: world               # "world" | "ee"
      #   gripper_mode: binary       # "none" | "binary" | "continuous_width"

Also set `env.views: len(image_topics)` and `env.act_dim` to whatever the chosen
action_type / gripper_mode produces (see _action_dim below).
"""

import bisect
import logging
import pickle
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import einops
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from .core import TrajectoryDataset

logger = logging.getLogger(__name__)

# All eef pose quaternions are XYZW per the bridge wire protocol.
_EEF_XYZ = slice(0, 3)
_EEF_QUAT = slice(3, 7)


class FactrPickleDataset(TrajectoryDataset):
    def __init__(
        self,
        data_directory: str,
        # Cameras (one or more, each becomes a view in the V dim)
        image_topics: Union[str, Sequence[str]] = (
            "/realsense/side/im",
            "/realsense/wrist/im",
        ),
        image_shape: Tuple[int, int] = (224, 224),
        # Action target
        # action_type: str = "eef_delta",
        action_type: str = "joint_absolute",
        joint_cmd_topic: str = "/factr_teleop/left/cmd_franka_pos",
        eef_pose_topic: str = "/franka/left/obs_franka_eef_pose",
        rot_unit: str = "axis",
        frame: str = "world",
        # Gripper handling (appended as last channel of action when enabled)
        gripper_mode: str = "binary",
        gripper_cmd_topic: str = "/factr_teleop/left/cmd_gripper_pos",
        gripper_width_topic: str = "/franka/left/obs_franka_gripper_width",
        gripper_open_below: float = 0.05,
        gripper_close_above: float = 0.40,
        # Optional proprioception (kept but not returned by get_frames)
        state_topic: Optional[str] = "/franka/left/obs_franka_torque",
        # Goals
        goal_dim: int = 3,
        goal_source: str = "future_eef",
        future_horizon: int = 30,
        # Misc
        prefetch: bool = False,
        subset_fraction: Optional[float] = None,
        episode_glob: str = "ep_*.pkl",
        seed: int = 42,
    ):
        if action_type not in ("joint_absolute", "joint_delta", "eef_absolute", "eef_delta"):
            raise ValueError(f"bad action_type: {action_type!r}")
        if rot_unit not in ("axis", "euler"):
            raise ValueError(f"bad rot_unit: {rot_unit!r}")
        if frame not in ("world", "ee"):
            raise ValueError(f"bad frame: {frame!r}")
        if gripper_mode not in ("none", "binary", "continuous_width"):
            raise ValueError(f"bad gripper_mode: {gripper_mode!r}")
        if goal_source not in ("none", "zero", "future_eef", "final_eef", "future_action"):
            raise ValueError(f"bad goal_source: {goal_source!r}")

        if isinstance(image_topics, str):
            image_topics = [image_topics]
        self.image_topics: List[str] = list(image_topics)

        self.dataset_root = Path(data_directory)
        self.image_shape = tuple(image_shape)
        self.action_type = action_type
        self.joint_cmd_topic = joint_cmd_topic
        self.eef_pose_topic = eef_pose_topic
        self.rot_unit = rot_unit
        self.frame = frame
        self.gripper_mode = gripper_mode
        self.gripper_cmd_topic = gripper_cmd_topic
        self.gripper_width_topic = gripper_width_topic
        self.gripper_open_below = gripper_open_below
        self.gripper_close_above = gripper_close_above
        self.state_topic = state_topic
        self.goal_dim = goal_dim
        self.goal_source = goal_source
        self.future_horizon = future_horizon
        self.prefetch_images = prefetch

        pkl_paths = sorted(self.dataset_root.glob(episode_glob))
        if not pkl_paths:
            raise FileNotFoundError(f"No episodes matching {episode_glob!r} under {self.dataset_root}")
        if subset_fraction is not None:
            assert 0 < subset_fraction <= 1
            n = max(1, int(round(len(pkl_paths) * subset_fraction)))
            pkl_paths = pkl_paths[:n]

        # Per-episode buffers, indexed by trajectory id (0..N-1)
        self._actions: List[np.ndarray] = []  # (T, action_dim) — training target
        self._eef_pose: List[np.ndarray] = []  # (T, 7) absolute [xyz + xyzw quat] — goals
        self._gripper: List[Optional[np.ndarray]] = []  # (T,) float32 in [0, 1], or None
        self._states: List[Optional[np.ndarray]] = []  # (T, state_dim) or None
        # Each entry of _jpegs is a list of length V; each view is a length-T list of JPEG buffers.
        self._jpegs: List[List[List[np.ndarray]]] = []
        self._decoded: List[Optional[np.ndarray]] = []  # (T, V, H, W, 3) uint8 if prefetch
        self._lengths: List[int] = []

        del seed  # reserved for future jitter
        for path in pkl_paths:
            self._load_episode(path)

        self.action_dim = self._actions[0].shape[1] if self._actions else 0
        logger.info(
            "FactrPickleDataset: %d episodes, V=%d, action_dim=%d, action_type=%s, gripper_mode=%s, goal_source=%s",
            len(self._lengths),
            len(self.image_topics),
            self.action_dim,
            self.action_type,
            self.gripper_mode,
            self.goal_source,
        )

    # ------------------------------------------------------------------ loading
    def _load_episode(self, path: Path) -> None:
        with path.open("rb") as f:
            episode = pickle.load(f)
        data = episode["data"]
        ts = episode["timestamps"]

        # Required topics depend on which signals we use.
        required = list(self.image_topics) + [self.eef_pose_topic]
        if self.action_type in ("joint_absolute", "joint_delta"):
            required.append(self.joint_cmd_topic)
        if self.gripper_mode == "binary":
            required.append(self.gripper_cmd_topic)
        elif self.gripper_mode == "continuous_width":
            required.append(self.gripper_width_topic)
        for k in required:
            if k not in data:
                raise KeyError(f"{path.name}: missing topic {k!r}")

        # Pick the first image topic as the canonical clock; other views aligned to it.
        primary_topic = self.image_topics[0]
        img_ts = np.asarray(ts[primary_topic], dtype=np.int64)

        # Per-view JPEG buffers, aligned to primary clock.
        per_view_jpegs: List[List[np.ndarray]] = []
        for topic in self.image_topics:
            if topic == primary_topic:
                bufs = [np.asarray(b, dtype=np.uint8) for b in data[topic]]
            else:
                other_ts = np.asarray(ts[topic], dtype=np.int64)
                idx = _nearest_indices(img_ts, other_ts)
                bufs = [np.asarray(data[topic][i], dtype=np.uint8) for i in idx]
            per_view_jpegs.append(bufs)

        # EEF pose (absolute) at image timestamps.
        eef_ts = np.asarray(ts[self.eef_pose_topic], dtype=np.int64)
        eef_idx = _nearest_indices(img_ts, eef_ts)
        eef_pose = np.stack(
            [np.asarray(data[self.eef_pose_topic][i], dtype=np.float32) for i in eef_idx],
            axis=0,
        )  # (T, 7)

        # Action component before gripper concat.
        if self.action_type in ("joint_absolute", "joint_delta"):
            cmd_ts = np.asarray(ts[self.joint_cmd_topic], dtype=np.int64)
            cmd_idx = _nearest_indices(img_ts, cmd_ts)
            joints = np.stack(
                [np.asarray(data[self.joint_cmd_topic][i], dtype=np.float32) for i in cmd_idx],
                axis=0,
            )  # (T, 7)
            if self.action_type == "joint_delta":
                arm = _step_delta(joints)
            else:
                arm = joints
        elif self.action_type == "eef_absolute":
            arm = _eef_pose_to_xyz_rot(eef_pose, self.rot_unit)  # (T, 6)
        else:  # eef_delta
            arm = _eef_pose_to_step_delta(eef_pose, self.rot_unit, self.frame)  # (T, 6)

        # Gripper signal as float in [0, 1] (binary 0/1 or scaled width).
        gripper = self._build_gripper(data, ts, img_ts)
        if gripper is not None:
            actions = np.concatenate([arm, gripper[:, None]], axis=1).astype(np.float32)
        else:
            actions = arm.astype(np.float32)

        # Optional proprioception.
        states: Optional[np.ndarray] = None
        if self.state_topic is not None and self.state_topic in data:
            st_ts = np.asarray(ts[self.state_topic], dtype=np.int64)
            st_idx = _nearest_indices(img_ts, st_ts)
            states = np.stack(
                [np.asarray(data[self.state_topic][i], dtype=np.float32) for i in st_idx],
                axis=0,
            )

        # Truncate all per-frame arrays to the same T (image-clock length).
        T = min(len(per_view_jpegs[0]), len(actions), len(eef_pose))
        per_view_jpegs = [v[:T] for v in per_view_jpegs]
        actions = actions[:T]
        eef_pose = eef_pose[:T]
        if gripper is not None:
            gripper = gripper[:T]
        if states is not None:
            states = states[:T]

        decoded: Optional[np.ndarray] = None
        if self.prefetch_images:
            decoded = np.stack(
                [np.stack([self._decode_jpeg(v[t]) for v in per_view_jpegs], axis=0) for t in range(T)],
                axis=0,
            )  # (T, V, H, W, 3) uint8

        self._jpegs.append(per_view_jpegs)
        self._actions.append(actions)
        self._eef_pose.append(eef_pose)
        self._gripper.append(gripper)
        self._states.append(states)
        self._decoded.append(decoded)
        self._lengths.append(T)

    def _build_gripper(self, data, ts, img_ts) -> Optional[np.ndarray]:
        if self.gripper_mode == "none":
            return None
        if self.gripper_mode == "binary":
            cg_ts = np.asarray(ts[self.gripper_cmd_topic], dtype=np.int64)
            cg_idx = _nearest_indices(img_ts, cg_ts)
            cg = np.stack(
                [np.asarray(data[self.gripper_cmd_topic][i], dtype=np.float32).reshape(-1)[0] for i in cg_idx],
                axis=0,
            )
            return _hysteresis(cg, self.gripper_open_below, self.gripper_close_above)
        # continuous_width
        gw_ts = np.asarray(ts[self.gripper_width_topic], dtype=np.int64)
        gw_idx = _nearest_indices(img_ts, gw_ts)
        gw = np.stack(
            [np.asarray(data[self.gripper_width_topic][i], dtype=np.float32).reshape(-1)[0] for i in gw_idx],
            axis=0,
        )
        # Franka Hand max width ≈ 0.08 m; scale to ~[0, 1].
        return np.clip(gw / 0.08, 0.0, 1.0).astype(np.float32)

    def _decode_jpeg(self, buf: np.ndarray) -> np.ndarray:
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode failed on JPEG buffer")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = self.image_shape
        if (img.shape[0], img.shape[1]) != (H, W):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        return img  # (H, W, 3) uint8 RGB

    # --------------------------------------------------------- TrajectoryDataset
    def __len__(self) -> int:
        return len(self._lengths)

    def get_seq_length(self, idx: int) -> int:
        return self._lengths[idx]

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(np.concatenate(self._actions, axis=0))

    def get_frames(
        self,
        idx: int,
        frames: Union[int, Sequence[int], range, np.ndarray],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(frames, int):
            frames = [frames]
        frames = np.asarray(list(frames), dtype=np.int64)
        if len(frames) == 0:
            raise ValueError("Requested 0 frames from FactrPickleDataset!")

        T = self._lengths[idx]
        if frames.min() < 0 or frames.max() >= T:
            raise IndexError(f"frame indices out of range for ep {idx} (T={T}): [{frames.min()}, {frames.max()}]")

        decoded = self._decoded[idx]
        if decoded is not None:
            imgs_np = decoded[frames]  # (k, V, H, W, 3) uint8
        else:
            views = self._jpegs[idx]
            imgs_np = np.stack(
                [np.stack([self._decode_jpeg(v[t]) for v in views], axis=0) for t in frames],
                axis=0,
            )  # (k, V, H, W, 3) uint8

        obs = torch.from_numpy(imgs_np).float() / 255.0  # (k, V, H, W, 3)
        obs = einops.rearrange(obs, "T V H W C -> T V C H W")

        actions = torch.from_numpy(self._actions[idx][frames]).float()  # (k, A)
        dummy_goal = torch.ones([obs.shape[0], 1, 1, 1])
        return obs, actions, dummy_goal

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self._lengths[idx]))

    # ----------------------------------------------------------------- helpers
    def _build_goals(self, idx: int, frames: np.ndarray) -> torch.Tensor:
        k = len(frames)
        if self.goal_source == "none":
            # Dummy (k, V=1, P=1, E=1) tensor for the "no goal" path in online_eval.py
            # (mirrors datasets/pusht.py). The codebase still requires a goal channel
            # because the rearrange("N T V P E -> N T (V P) E") assumes 5-D after batching.
            return torch.ones((k, 1, 1, 1), dtype=torch.float32)
        if self.goal_source == "zero":
            return torch.zeros((k, self.goal_dim), dtype=torch.float32)

        T = self._lengths[idx]
        goals_np = np.zeros((k, self.goal_dim), dtype=np.float32)

        if self.goal_source == "future_eef":
            xyz = self._eef_pose[idx][:, _EEF_XYZ]
            future_idx = np.clip(frames + self.future_horizon, 0, T - 1)
            slice_dim = min(self.goal_dim, 3)
            goals_np[:, :slice_dim] = xyz[future_idx, :slice_dim]
        elif self.goal_source == "final_eef":
            xyz = self._eef_pose[idx][:, _EEF_XYZ]
            slice_dim = min(self.goal_dim, 3)
            goals_np[:, :slice_dim] = xyz[T - 1, :slice_dim]
        else:  # future_action (legacy)
            future_idx = np.clip(frames + self.future_horizon, 0, T - 1)
            actions = self._actions[idx]
            slice_dim = min(self.goal_dim, actions.shape[1])
            goals_np[:, :slice_dim] = actions[future_idx, :slice_dim]
        return torch.from_numpy(goals_np)


# --------------------------------------------------------------------- utils
def _nearest_indices(query_ts: np.ndarray, ref_ts: np.ndarray) -> np.ndarray:
    """For each q in query_ts, return the index into ref_ts of the nearest timestamp.

    Both arrays must be sorted ascending. O((len(q) + len(r)) log len(r)).
    """
    n = len(ref_ts)
    if n == 0:
        raise ValueError("Reference timestamp array is empty")
    out = np.empty(len(query_ts), dtype=np.int64)
    for i, q in enumerate(query_ts):
        j = bisect.bisect_left(ref_ts, q)
        if j == 0:
            out[i] = 0
        elif j >= n:
            out[i] = n - 1
        else:
            out[i] = j if (ref_ts[j] - q) < (q - ref_ts[j - 1]) else j - 1
    return out


def _step_delta(x: np.ndarray) -> np.ndarray:
    """a_t = x_{t+1} - x_t, with the last frame repeating the prior delta."""
    if len(x) < 2:
        return np.zeros_like(x)
    d = np.diff(x, axis=0)
    return np.concatenate([d, d[-1:]], axis=0)


def _eef_pose_to_xyz_rot(pose: np.ndarray, rot_unit: str) -> np.ndarray:
    """(T,7) pose [xyz + quat XYZW] -> (T,6) [xyz + axis-angle or euler-xyz]."""
    xyz = pose[:, _EEF_XYZ]
    quat_xyzw = pose[:, _EEF_QUAT]
    rot = R.from_quat(quat_xyzw)
    if rot_unit == "axis":
        rotvec = rot.as_rotvec()
    else:
        rotvec = rot.as_euler("xyz", degrees=False)
    return np.concatenate([xyz, rotvec], axis=1).astype(np.float32)


def _eef_pose_to_step_delta(pose: np.ndarray, rot_unit: str, frame: str) -> np.ndarray:
    """(T,7) pose -> (T,6) per-step Cartesian delta. Last frame repeats the prior delta.

    frame="world": dxyz = xyz_{t+1} - xyz_t; rotation delta = R_{t+1} R_t^-1.
    frame="ee":    dxyz = R_t^-1 (xyz_{t+1} - xyz_t); rotation delta = R_t^-1 R_{t+1}.
    """
    T = len(pose)
    out = np.zeros((T, 6), dtype=np.float32)
    if T < 2:
        return out
    rots = R.from_quat(pose[:, _EEF_QUAT])
    xyz = pose[:, _EEF_XYZ]
    dxyz_w = xyz[1:] - xyz[:-1]  # (T-1, 3)
    r_t = rots[:-1]
    r_tp1 = rots[1:]
    if frame == "world":
        dxyz = dxyz_w
        r_delta = r_tp1 * r_t.inv()
    else:  # ee
        dxyz = r_t.inv().apply(dxyz_w)
        r_delta = r_t.inv() * r_tp1
    if rot_unit == "axis":
        drot = r_delta.as_rotvec()
    else:
        drot = r_delta.as_euler("xyz", degrees=False)
    out[:-1, :3] = dxyz
    out[:-1, 3:] = drot
    out[-1] = out[-2]  # repeat last delta
    return out


def _hysteresis(cg: np.ndarray, open_below: float, close_above: float) -> np.ndarray:
    """Binary gripper from leader trigger angle, mimicking the NUC bridge."""
    state = 0  # 0=open, 1=closed
    out = np.empty(len(cg), dtype=np.float32)
    for i, x in enumerate(cg):
        if state == 0 and x > close_above:
            state = 1
        elif state == 1 and x < open_below:
            state = 0
        out[i] = float(state)
    return out
