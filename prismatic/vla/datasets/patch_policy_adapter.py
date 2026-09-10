"""
patch_policy_adapter.py

Adapter that wraps a TrajectorySlicerDataset (from patch_policy) and converts each sample
into the dict format expected by OpenVLA's forward pass and PaddedCollatorForActionPrediction.
"""
import random
from typing import Optional, Type

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder, QwenPromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import IGNORE_INDEX, NUM_TOKENS


def _tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    """Convert a (C, H, W) float [0,1] tensor to a PIL Image."""
    img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(img_np)


class PatchPolicyVLADataset(Dataset):
    """
    Wraps a TrajectorySlicerDataset and converts its output into the dict format
    expected by OpenVLA's PaddedCollatorForActionPrediction.

    The TrajectorySlicerDataset returns (obs, actions, goal/mask) where:
        - obs: (T, V, C, H, W) float [0,1]
        - actions: (T, action_dim) float
        - goal/mask: varies per dataset

    This adapter:
        1. Takes the first frame as the primary image
        2. Optionally takes a second view as the wrist image
        3. Builds the VLA prompt with language instruction
        4. Tokenizes actions into the VLA's action token format
        5. Normalizes actions to [-1, 1] using q01/q99 statistics
        6. Returns a dict compatible with PaddedCollatorForActionPrediction
    """

    def __init__(
        self,
        trajectory_slicer_dataset,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
        language_instruction: Optional[str],
        num_actions_chunk: int,
        action_dim: int,
        use_wrist_image: bool = False,
        use_proprio: bool = False,
        predict_stop_token: bool = True,
        dataset_name: str = "patch_policy_dataset",
        use_minivlm: bool = False,
    ):
        self.slicer = trajectory_slicer_dataset
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn
        self.language_instruction = language_instruction
        self.num_actions_chunk = num_actions_chunk
        self.action_dim = action_dim
        self.use_wrist_image = use_wrist_image
        self.use_proprio = use_proprio
        self.predict_stop_token = predict_stop_token
        self.dataset_name = dataset_name
        self.use_minivlm = use_minivlm

        # Compute action statistics for normalization to [-1, 1]
        self._dataset_statistics = None
        self._action_q01 = None
        self._action_q99 = None
        self._compute_action_stats()

    def _compute_action_stats(self):
        """Compute q01/q99 action statistics from the underlying dataset for normalization."""
        print("Computing action statistics for normalization...")
        all_actions = self.slicer.get_all_actions()
        if isinstance(all_actions, torch.Tensor):
            all_actions = all_actions.numpy()
        self._action_q01 = np.quantile(all_actions, 0.01, axis=0).astype(np.float32)
        self._action_q99 = np.quantile(all_actions, 0.99, axis=0).astype(np.float32)
        self._action_mean = np.mean(all_actions, axis=0).astype(np.float32)
        self._action_std = np.std(all_actions, axis=0).astype(np.float32)
        self._action_min = np.min(all_actions, axis=0).astype(np.float32)
        self._action_max = np.max(all_actions, axis=0).astype(np.float32)
        print(f"  action q01: {self._action_q01}")
        print(f"  action q99: {self._action_q99}")

        self._dataset_statistics = {
            self.dataset_name: {
                "action": {
                    "q01": self._action_q01,
                    "q99": self._action_q99,
                    "mean": self._action_mean,
                    "std": self._action_std,
                    "min": self._action_min,
                    "max": self._action_max,
                }
            }
        }

    def _normalize_actions(self, actions: np.ndarray) -> np.ndarray:
        """Normalize actions to [-1, 1] using q01/q99 bounds."""
        q01 = self._action_q01
        q99 = self._action_q99
        # Avoid division by zero
        scale = np.where(q99 - q01 > 1e-8, q99 - q01, 1.0)
        normalized = 2.0 * (actions - q01) / scale - 1.0
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)

    @property
    def dataset_statistics(self):
        return self._dataset_statistics

    def _get_language_instruction(self, idx):
        """Get language instruction for a given sample index."""
        if self.language_instruction is not None:
            return self.language_instruction

        # For LIBERO: extract task name from demo directory structure
        underlying_dataset = self.slicer.dataset
        # Walk through subset wrappers to find the actual dataset
        while hasattr(underlying_dataset, 'dataset'):
            underlying_dataset = underlying_dataset.dataset

        if hasattr(underlying_dataset, 'get_task_name_for_demo'):
            # Get the trajectory index from the slicer's slice info
            traj_idx, _, _ = self.slicer.slices[idx]
            # If wrapped in TrajectorySubset, map to original index
            dataset = self.slicer.dataset
            if hasattr(dataset, 'indices'):
                traj_idx = dataset.indices[traj_idx]
            return underlying_dataset.get_task_name_for_demo(traj_idx)

        return "complete the task"

    def __len__(self):
        return len(self.slicer)

    def __getitem__(self, idx):
        # 1. Get (obs, actions, goal/mask) from trajectory slicer
        sample = self.slicer[idx]
        obs, actions = sample[0], sample[1]
        # obs: (T, V, C, H, W) in [0,1]
        # actions: (T, action_dim)

        # 2. Take first frame as primary image, convert to PIL and apply transform
        primary_img = _tensor_to_pil(obs[0, 0])  # First timestep, first view
        pixel_values = self.image_transform(primary_img)

        # 3. If multi-view, get wrist/secondary image
        pixel_values_wrist = None
        if self.use_wrist_image and obs.shape[1] > 1:
            wrist_img = _tensor_to_pil(obs[0, 1])  # First timestep, second view
            pixel_values_wrist = self.image_transform(wrist_img)

        # 4. Build action chunk from first num_actions_chunk timesteps
        if isinstance(actions, torch.Tensor):
            action_chunk = actions[:self.num_actions_chunk].numpy()
        else:
            action_chunk = actions[:self.num_actions_chunk]
        # Pad if we have fewer actions than num_actions_chunk
        if len(action_chunk) < self.num_actions_chunk:
            pad_size = self.num_actions_chunk - len(action_chunk)
            action_chunk = np.concatenate([action_chunk, np.zeros((pad_size, self.action_dim), dtype=np.float32)])

        # 5. Normalize actions to [-1, 1] for tokenization
        action_chunk_normalized = self._normalize_actions(action_chunk)

        # 6. Get language instruction
        lang = self._get_language_instruction(idx)

        # 7./8. Tokenize prompt + build input_ids / labels
        current_action = action_chunk_normalized[0]
        future_actions = action_chunk_normalized[1:]

        if self.use_minivlm:
            # VLA-Adapter (Qwen2.5 backbone): the prompt is followed by NUM_TOKENS action-query placeholder tokens.
            # Mirrors RLDSBatchTransform.__call__ (use_minivlm branch) in prismatic/vla/datasets/datasets.py.
            prompt_builder = QwenPromptBuilder("openvla")
            future_actions_string = self.action_tokenizer(future_actions, self.use_minivlm)
            current_action_string = self.action_tokenizer(current_action, self.use_minivlm)
            action_chunk_string = [current_action_string] + future_actions_string
            flattened_action_chunk_string = [item for sublist in action_chunk_string for item in sublist]

            conversation = [
                {"from": "human", "value": f"What action should the robot take to {lang}?"},
                {"from": "gpt", "value": ""},
            ]
            for turn in conversation:
                prompt_builder.add_turn(turn["from"], turn["value"])
            input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
            if len(input_ids) >= 3:
                del input_ids[-3]
                del input_ids[-2]
                del input_ids[-1]

            if NUM_TOKENS < len(flattened_action_chunk_string):
                input_ids = input_ids + flattened_action_chunk_string[:NUM_TOKENS]
            else:
                remaining_length = NUM_TOKENS - len(flattened_action_chunk_string)
                extended_array = random.choices(flattened_action_chunk_string, k=remaining_length)
                input_ids = input_ids + flattened_action_chunk_string + extended_array
            labels = list(input_ids)
            action_chunk_len = NUM_TOKENS
        else:
            prompt_builder = self.prompt_builder_fn("openvla")
            current_action_string = self.action_tokenizer(current_action)
            future_actions_string = ''.join(self.action_tokenizer(future_actions))
            action_chunk_string = current_action_string + future_actions_string
            action_chunk_len = len(action_chunk_string)

            conversation = [
                {"from": "human", "value": f"What action should the robot take to {lang}?"},
                {"from": "gpt", "value": action_chunk_string},
            ]
            for turn in conversation:
                prompt_builder.add_turn(turn["from"], turn["value"])
            input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
            labels = list(input_ids)

        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)

        # Mask everything except the action tokens in labels
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        # 9. Return dict matching PaddedCollatorForActionPrediction expectations
        return_dict = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            actions=action_chunk_normalized,  # normalized actions for L1/diffusion loss
        )
        if self.use_wrist_image and pixel_values_wrist is not None:
            return_dict["pixel_values_wrist"] = pixel_values_wrist

        return return_dict
