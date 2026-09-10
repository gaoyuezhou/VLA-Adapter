"""
Important constants for VLA training and evaluation.

Constants can be set explicitly via set_constants() (used by Hydra configs),
or auto-detected from command line arguments (legacy Draccus behavior).
"""
import sys
from enum import Enum

# Qwen2.5-0.5B token constants
IGNORE_INDEX = -100
# ACTION_TOKEN_BEGIN_IDX = 31743
ACTION_TOKEN_BEGIN_IDX  = 151386
STOP_INDEX = 2  # '</s>'
NUM_TOKENS = 64


# Defines supported normalization schemes for action and proprioceptive state.
class NormalizationType(str, Enum):
    # fmt: off
    NORMAL = "normal"               # Normalize to Mean = 0, Stdev = 1
    BOUNDS = "bounds"               # Normalize to Interval = [-1, 1]
    BOUNDS_Q99 = "bounds_q99"       # Normalize [quantile_01, ..., quantile_99] --> [-1, ..., 1]
    # fmt: on


# Define constants for each robot platform
LIBERO_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

CALVIN_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

ALOHA_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 25,
    "ACTION_DIM": 14,
    "PROPRIO_DIM": 14,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,
}

BRIDGE_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 5,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 7,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}


def set_constants(action_dim, num_actions_chunk, proprio_dim, normalization_type):
    """Set robot constants explicitly from config. Must be called before training begins."""
    global ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE
    ACTION_DIM = action_dim
    NUM_ACTIONS_CHUNK = num_actions_chunk
    PROPRIO_DIM = proprio_dim
    if isinstance(normalization_type, str):
        ACTION_PROPRIO_NORMALIZATION_TYPE = NormalizationType(normalization_type)
    else:
        ACTION_PROPRIO_NORMALIZATION_TYPE = normalization_type
    print(f"Constants set via config:")
    print(f"  NUM_ACTIONS_CHUNK = {NUM_ACTIONS_CHUNK}")
    print(f"  ACTION_DIM = {ACTION_DIM}")
    print(f"  PROPRIO_DIM = {PROPRIO_DIM}")
    print(f"  ACTION_PROPRIO_NORMALIZATION_TYPE = {ACTION_PROPRIO_NORMALIZATION_TYPE}")


# Function to detect robot platform from command line arguments (legacy fallback)
def detect_robot_platform():
    cmd_args = " ".join(sys.argv).lower()

    if "libero" in cmd_args:
        return "LIBERO"
    elif "aloha" in cmd_args:
        return "ALOHA"
    elif "bridge" in cmd_args:
        return "BRIDGE"
    elif "calvin" in cmd_args:
        return "CALVIN"
    else:
        # Default to LIBERO if unclear
        return "LIBERO"


# Determine which robot platform to use (default; can be overridden by set_constants())
ROBOT_PLATFORM = detect_robot_platform()

# Set the appropriate constants based on the detected platform
if ROBOT_PLATFORM == "LIBERO":
    _defaults = LIBERO_CONSTANTS
elif ROBOT_PLATFORM == "ALOHA":
    _defaults = ALOHA_CONSTANTS
elif ROBOT_PLATFORM == "BRIDGE":
    _defaults = BRIDGE_CONSTANTS
elif ROBOT_PLATFORM == "CALVIN":
    _defaults = CALVIN_CONSTANTS

# Assign constants to global variables (these can be overridden by set_constants())
NUM_ACTIONS_CHUNK = _defaults["NUM_ACTIONS_CHUNK"]
ACTION_DIM = _defaults["ACTION_DIM"]
PROPRIO_DIM = _defaults["PROPRIO_DIM"]
ACTION_PROPRIO_NORMALIZATION_TYPE = _defaults["ACTION_PROPRIO_NORMALIZATION_TYPE"]

# Print which robot platform constants are being used (for debugging)
print(f"Using {ROBOT_PLATFORM} constants (default, may be overridden by config):")
print(f"  NUM_ACTIONS_CHUNK = {NUM_ACTIONS_CHUNK}")
print(f"  ACTION_DIM = {ACTION_DIM}")
print(f"  PROPRIO_DIM = {PROPRIO_DIM}")
print(f"  ACTION_PROPRIO_NORMALIZATION_TYPE = {ACTION_PROPRIO_NORMALIZATION_TYPE}")