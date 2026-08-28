"""The fixed RefSight training configuration is defined here."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_TRAINING_ROOT = Path(__file__).resolve().parent


def _path_from_environment(name: str, default: Path) -> Path:
    """A path is read from the environment or from the default."""
    value = os.getenv(name)
    return Path(value).expanduser() if value else default


# Extracted clips are expected in this directory.
OUTPUT_DIR = _path_from_environment(
    "REFSIGHT_TIME_SERIES_DIR",
    PROJECT_ROOT / "data" / "time_series_clips",
)
RESULTS_DIR = _path_from_environment(
    "OVERRIDE_RESULTS_DIR",
    _path_from_environment(
        "REFSIGHT_RESULTS_DIR",
        MODEL_TRAINING_ROOT / "results" / "single_run",
    ),
)
SEQ_DIR = OUTPUT_DIR / "sequences"
META_CSV = OUTPUT_DIR / "metadata.csv"
SCHEMA_JSON = OUTPUT_DIR / "schema.json"

# The 32 model input features are listed in their stored order.
MODEL_FEATURE_COLUMNS = (
    "referee_speed",
    "ball_height",
    "ball_speed",
    "referee_to_ball_distance",
    "interaction_angle",
    "occlusion_count",
    "distance_covered_previous_5s",
    "trajectory_directness_5s",
    "fastest_player_speed",
    "closest_player_speed",
    "turn_angle",
    "trajectory_directness_2s",
    "lateral_view_angle",
    "referee_to_lead_assistant_distance",
    "ball_angle_referee_vs_lead_assistant",
    "complementary_coverage_index",
    "home_team_centroid_x",
    "home_team_centroid_y",
    "home_team_length",
    "home_team_width",
    "home_team_stretch_index",
    "home_team_spread",
    "home_team_surface_area",
    "home_team_centroid_to_ball_distance",
    "away_team_centroid_x",
    "away_team_centroid_y",
    "away_team_length",
    "away_team_width",
    "away_team_stretch_index",
    "away_team_spread",
    "away_team_surface_area",
    "away_team_centroid_to_ball_distance",
)

# Sequence and optimisation settings are fixed across model types.
FPS = 25
TGT_LEN = 375
MIN_CLIP_SECONDS = 2
MIN_CLIP_FRAMES = FPS * MIN_CLIP_SECONDS
BATCH_SIZE = 16
EPOCHS = 50
LR = 3e-4
WEIGHT_DECAY = 5e-3
GRADIENT_CLIP_NORM = 5.0

# LSTM settings are defined here.
HIDDEN = 64
LAYERS = 2
BIDIR = True
DROPOUT = 0.1

# TCN settings are defined here.
TCN_CHANNELS = (64, 64, 64, 64)
TCN_KERNEL = 5
TCN_DROPOUT = 0.5

# Transformer settings are defined here.
TRANSFORMER_D_MODEL = 128
TRANSFORMER_NHEAD = 8
TRANSFORMER_NUM_LAYERS = 4
TRANSFORMER_DIM_FF = 256
TRANSFORMER_DROPOUT = 0.1

# The referee-level partition is fixed across all model types and seeds.
REFEREE_COL = "Referee Name"
N_TEST_REFEREES = 5
# Whole referee groups are added until this minimum is reached.
MINIMUM_VALIDATION_QUERIES = 300
SPLIT_SEED = 42
EXPECTED_QUERY_COUNTS = {"train": 989, "validation": 374, "test": 486}
EXPECTED_TOTAL_QUERIES = sum(EXPECTED_QUERY_COUNTS.values())

# Only weight initialisation and minibatch order are changed between seeds.
SEED = int(os.getenv("OVERRIDE_SEED", os.getenv("REFSIGHT_SEED", "42")))

# One of the three evaluated architectures is selected for each run.
MODEL_NAME = os.getenv(
    "OVERRIDE_MODEL_NAME",
    os.getenv("REFSIGHT_MODEL_NAME", "transformer"),
).lower()
if MODEL_NAME not in {"lstm", "tcn", "transformer"}:
    raise ValueError("The model name must be lstm, tcn, or transformer.")

# Each matched list is formed from one incorrect clip and seven correct clips.
TOP_K_CORRECT = 7
LISTWISE_TAU = 0.5

# Near-constant training features are kept numerically stable.
MIN_STD = 1e-6
