# Feature engineering

This folder converts framewise tracking data into the spatiotemporal features
used by RefSight. The feature names are consistent with the paper and
supplementary material.

No tracking data or featured output is included. The source tracking data is
controlled by the data provider. It must be obtained under the provider's terms.
Raw and featured files should be kept outside the public repository.

## Files

- `features.py` contains the feature definitions.
- `run_all.py` processes a folder of match CSV files.
- `requirements.txt` contains the Python dependencies.

## Feature set

The following public features are copied from source tracking measurements:

- `referee_speed`
- `ball_height`
- `ball_speed`

The following variables are created from the tracking data:

- Referee movement: `distance_covered_previous_5s`,
  `trajectory_directness_5s`, `trajectory_directness_2s`, and `turn_angle`.
- Referee position and viewing: `referee_to_ball_distance`,
  `interaction_angle`, `occlusion_count`, and `lateral_view_angle`.
- Officiating-team coverage: `referee_to_lead_assistant_distance`,
  `ball_angle_referee_vs_lead_assistant`, and
  `complementary_coverage_index`.
- Player speed: `fastest_player_speed` and `closest_player_speed`.
- Home team organisation: `home_team_centroid_x`, `home_team_centroid_y`,
  `home_team_length`, `home_team_width`, `home_team_stretch_index`,
  `home_team_spread`, `home_team_surface_area`, and
  `home_team_centroid_to_ball_distance`.
- Away team organisation: `away_team_centroid_x`, `away_team_centroid_y`,
  `away_team_length`, `away_team_width`, `away_team_stretch_index`,
  `away_team_spread`, `away_team_surface_area`, and
  `away_team_centroid_to_ball_distance`.

The source fields `Referee_speed` and `ball_z` are exposed as
`referee_speed` and `ball_height`. This translation is applied during feature
engineering. Downstream code uses only the public feature names.

| Paper feature | Public column |
|---|---|
| Referee speed | `referee_speed` |
| Ball height | `ball_height` |
| Ball speed | `ball_speed` |
| Referee to ball distance | `referee_to_ball_distance` |
| Interaction angle | `interaction_angle` |
| Occlusion count | `occlusion_count` |
| Distance covered (previous 5 s) | `distance_covered_previous_5s` |
| Trajectory directness (5 s) | `trajectory_directness_5s` |
| Fastest player speed | `fastest_player_speed` |
| Closest player speed | `closest_player_speed` |
| Turn angle | `turn_angle` |
| Trajectory directness (2 s) | `trajectory_directness_2s` |
| Lateral view angle | `lateral_view_angle` |
| Referee to lead assistant distance | `referee_to_lead_assistant_distance` |
| Ball angle (referee vs lead assistant) | `ball_angle_referee_vs_lead_assistant` |
| Complementary coverage index | `complementary_coverage_index` |
| Home team centroid, x coordinate | `home_team_centroid_x` |
| Home team centroid, y coordinate | `home_team_centroid_y` |
| Home team length | `home_team_length` |
| Home team width | `home_team_width` |
| Home team stretch index | `home_team_stretch_index` |
| Home team spread | `home_team_spread` |
| Home team surface area | `home_team_surface_area` |
| Home team centroid to ball distance | `home_team_centroid_to_ball_distance` |
| Away team centroid, x coordinate | `away_team_centroid_x` |
| Away team centroid, y coordinate | `away_team_centroid_y` |
| Away team length | `away_team_length` |
| Away team width | `away_team_width` |
| Away team stretch index | `away_team_stretch_index` |
| Away team spread | `away_team_spread` |
| Away team surface area | `away_team_surface_area` |
| Away team centroid to ball distance | `away_team_centroid_to_ball_distance` |

The goalkeeper is excluded from each team-organisation calculation. The
goalkeeper slot is inferred from the mean distance to either goal.

Intermediate geometry values are kept within the feature functions. They are
not added to the output table.

## Input format

One match is stored in each CSV file. One row represents one frame. The sampling
rate is 25 Hz. The table is stored in wide format.

The following fields are required:

- `live`
- `period` or `gamePeriod`
- `ball_x`, `ball_y`, `ball_z`, and `ball_speed`
- `Referee_x`, `Referee_y`, and `Referee_speed`
- `AsstRef1_x`, `AsstRef1_y`, `AsstRef2_x`, and `AsstRef2_y`
- `pitchLength`
- `lastTouch`
- `home_1_x`, `home_1_y`, and `home_1_speed` through slot 11
- `away_1_x`, `away_1_y`, and `away_1_speed` through slot 11

Coordinates are measured in metres. Speeds are measured in metres per second.
The pitch centre is the coordinate origin.

The `live` field may use Boolean values, `0` and `1`, or common text values such
as `true` and `false`. Unsupported values cause a clear error.

The following source fields are preserved for DTW retrieval and model-data
partitioning:

- `MergeID`
- `frameIdx`
- `matchSeconds`
- `Time`
- `event`
- `Incorrect_Decision`
- `EventName`
- `Type`
- `Referee Name`

These fields must be present when the complete RefSight workflow is run.

## Installation

Python 3.11 or a later compatible version is recommended.
The following commands are run from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r "Feature Engineering/requirements.txt"
```

## Run

The default input folder is `data/raw_tracking`. The default output folder is
`data/featured_tracking`. Paths are resolved from the repository root.

```bash
python "Feature Engineering/run_all.py"
```

Other folders can be supplied on the command line:

```bash
python "Feature Engineering/run_all.py" \
  --input-dir /path/to/raw_tracking \
  --output-dir /path/to/featured_tracking
```

The same folders can be set with environment variables:

```bash
export REFSIGHT_RAW_TRACKING_DIR=/path/to/raw_tracking
export REFSIGHT_FEATURED_TRACKING_DIR=/path/to/featured_tracking
python "Feature Engineering/run_all.py"
```

Existing Parquet files are skipped. They are replaced only when `--overwrite`
is supplied.

Feature calculations are performed on the full match timeline. Live-play
boundaries are respected by time-window calculations. Only live-play rows are
written to each Parquet file.

## Data protection

Generated Parquet files may contain licensed tracking data and match metadata.
They must not be committed to the public repository. The root `.gitignore`
excludes Parquet files and the `data` folder.
