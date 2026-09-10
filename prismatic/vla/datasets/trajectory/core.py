"""
Trajectory dataset base classes and utilities.
Adapted from patch_policy/datasets/core.py
"""
import abc
import torch
import numpy as np
from torch import default_generator, randperm
from torch.utils.data import Dataset, Subset
from typing import Callable, Optional, Sequence, List, Any


def _accumulate(iterable, fn=lambda x, y: x + y):
    """Return running totals."""
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = fn(total, element)
        yield total


def _repeat_start_to_length(x: torch.Tensor, length: int, dim: int = 0):
    """Repeat the first frame to pad tensor to desired length along dim."""
    pad_size = length - x.shape[dim]
    if pad_size <= 0:
        return x
    first_frame = x.index_select(dim, torch.tensor(0, device=x.device))
    repeat_shape = [1] * len(x.shape)
    repeat_shape[dim] = pad_size
    pad = first_frame.repeat(*repeat_shape)
    return torch.cat([pad, x], dim=dim)


class TrajectoryDataset(Dataset, abc.ABC):
    """
    A dataset containing trajectories.
    TrajectoryDataset[i] returns: (observations, actions, mask)
        observations: Tensor[T, ...], T frames of observations
        actions: Tensor[T, ...], T frames of actions
        mask: Tensor[T]: False: invalid; True: valid
    """

    @abc.abstractmethod
    def get_seq_length(self, idx):
        raise NotImplementedError

    @abc.abstractmethod
    def get_frames(self, idx, frames):
        raise NotImplementedError


class TrajectorySubset(TrajectoryDataset, Subset):
    def __init__(self, dataset: TrajectoryDataset, indices: Sequence[int]):
        Subset.__init__(self, dataset, indices)

    def get_seq_length(self, idx):
        return self.dataset.get_seq_length(self.indices[idx])

    def get_all_actions(self):
        return self.dataset.get_all_actions()

    def get_frames(self, idx, frames):
        return self.dataset.get_frames(self.indices[idx], frames)


class TrajectorySlicerDataset:
    def __init__(
        self,
        dataset: TrajectoryDataset,
        window: int,
        future_conditional: bool = False,
        min_future_sep: int = 0,
        future_seq_len: Optional[int] = None,
        only_sample_tail: bool = False,
        transform: Optional[Callable] = None,
        num_extra_predicted_actions: Optional[int] = None,
        frame_step: int = 1,
        repeat_first_frame: bool = False,
    ):
        if future_conditional:
            assert future_seq_len is not None, "must specify a future_seq_len"
        self.dataset = dataset
        self.window = window
        self.future_conditional = future_conditional
        self.min_future_sep = min_future_sep
        self.future_seq_len = future_seq_len
        self.only_sample_tail = only_sample_tail
        self.transform = transform
        self.num_extra_predicted_actions = num_extra_predicted_actions or 0
        self.slices = []
        self.frame_step = frame_step
        min_seq_length = np.inf
        if num_extra_predicted_actions:
            window = window + num_extra_predicted_actions
        for i in range(len(self.dataset)):
            T = self.dataset.get_seq_length(i)
            min_seq_length = min(T, min_seq_length)
            if T - window < 0:
                print(f"Ignored short sequence #{i}: len={T}, window={window}")
            else:
                if repeat_first_frame:
                    self.slices += [(i, 0, end + 1) for end in range(window - 1)]
                window_len_with_step = (window - 1) * frame_step + 1
                last_start = T - window_len_with_step
                self.slices += [
                    (i, start, start + window_len_with_step)
                    for start in range(last_start)
                ]

        if min_seq_length < window:
            print(
                f"Ignored short sequences. To include all, set window <= {min_seq_length}."
            )

    def get_seq_length(self, idx: int) -> int:
        if self.future_conditional:
            return self.future_seq_len + self.window
        else:
            return self.window

    def get_all_actions(self) -> torch.Tensor:
        return self.dataset.get_all_actions()

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        i, start, end = self.slices[idx]
        T = self.dataset.get_seq_length(i)

        if (
            self.num_extra_predicted_actions is not None
            and self.num_extra_predicted_actions != 0
        ):
            assert self.frame_step == 1, "NOT TESTED"
            if self.future_conditional:
                raise NotImplementedError(
                    "num_extra_predicted_actions with future_conditional not implemented"
                )
            assert end <= T, f"end={end} > T={T}"
            observations, actions, mask = self.dataset.get_frames(i, range(start, end))
            observations = observations[: self.window]
            values = [observations, actions, mask.bool()]
        else:
            if self.future_conditional:
                assert self.frame_step == 1, "NOT TESTED"
                valid_start_range = (
                    end + self.min_future_sep,
                    self.dataset.get_seq_length(i) - self.future_seq_len,
                )
                if valid_start_range[0] < valid_start_range[1]:
                    if self.only_sample_tail:
                        future_obs_range = range(T - self.future_seq_len, T)
                    else:
                        future_start = np.random.randint(*valid_start_range)
                        future_end = future_start + self.future_seq_len
                        future_obs_range = range(future_start, future_end)
                    obs, actions, mask = self.dataset.get_frames(
                        i, list(range(start, end)) + list(future_obs_range)
                    )
                    future_obs = obs[end - start :]
                    obs = obs[: end - start]
                    actions = actions[: end - start]
                    mask = mask[: end - start]
                else:
                    obs, actions, mask = self.dataset.get_frames(i, range(start, end))
                    obs_dims = obs.shape[1:]
                    future_obs = torch.zeros((self.future_seq_len, *obs_dims))
                values = [obs, actions, mask.bool(), future_obs]
            else:
                observations, actions, mask = self.dataset.get_frames(
                    i, range(start, end, self.frame_step)
                )
                values = [observations, actions, mask.bool()]

        if end - start < self.window + self.num_extra_predicted_actions:
            values = [
                _repeat_start_to_length(
                    x, self.window + self.num_extra_predicted_actions, dim=0
                )
                for x in values
            ]
            values[0] = values[0][: self.window]

        if self.transform is not None:
            values = self.transform(values)
        return tuple(values)


def get_train_val_sliced(
    traj_dataset: TrajectoryDataset,
    train_fraction: float = 0.9,
    random_seed: int = 42,
    window_size: int = 10,
    future_conditional: bool = False,
    min_future_sep: int = 0,
    future_seq_len: Optional[int] = None,
    only_sample_tail: bool = False,
    transform: Optional[Callable[[Any], Any]] = None,
    num_extra_predicted_actions: Optional[int] = None,
    frame_step: int = 1,
):
    train, val = split_traj_datasets(
        traj_dataset,
        train_fraction=train_fraction,
        random_seed=random_seed,
    )
    traj_slicer_kwargs = {
        "window": window_size,
        "future_conditional": future_conditional,
        "min_future_sep": min_future_sep,
        "future_seq_len": future_seq_len,
        "only_sample_tail": only_sample_tail,
        "transform": transform,
        "num_extra_predicted_actions": num_extra_predicted_actions,
        "frame_step": frame_step,
    }
    train_slices = TrajectorySlicerDataset(train, **traj_slicer_kwargs)
    val_slices = TrajectorySlicerDataset(val, **traj_slicer_kwargs)
    return train_slices, val_slices


def random_split_traj(
    dataset: TrajectoryDataset,
    lengths: Sequence[int],
    generator: Optional[torch.Generator] = default_generator,
) -> List[TrajectorySubset]:
    if sum(lengths) != len(dataset):
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )
    indices = randperm(sum(lengths), generator=generator).tolist()
    return [
        TrajectorySubset(dataset, indices[offset - length : offset])
        for offset, length in zip(_accumulate(lengths), lengths)
    ]


def split_traj_datasets(dataset, train_fraction=0.95, random_seed=42):
    dataset_length = len(dataset)
    lengths = [
        int(train_fraction * dataset_length),
        dataset_length - int(train_fraction * dataset_length),
    ]
    train_set, val_set = random_split_traj(
        dataset, lengths, generator=torch.Generator().manual_seed(random_seed)
    )
    return train_set, val_set
