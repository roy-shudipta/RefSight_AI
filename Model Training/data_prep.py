"""Metadata, referee splits, matched lists, and scaling are prepared here."""

from __future__ import annotations

import json
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from config import (
    META_CSV,
    EXPECTED_QUERY_COUNTS,
    EXPECTED_TOTAL_QUERIES,
    MIN_CLIP_FRAMES,
    MIN_STD,
    MODEL_FEATURE_COLUMNS,
    N_TEST_REFEREES,
    REFEREE_COL,
    SCHEMA_JSON,
    SPLIT_SEED,
    TGT_LEN,
    TOP_K_CORRECT,
    MINIMUM_VALIDATION_QUERIES,
)
from utils import coerce_numeric, make_abs_path


def load_metadata() -> pd.DataFrame:
    """Clip metadata is loaded and checked."""
    if not META_CSV.exists():
        raise FileNotFoundError(f"Metadata was not found at {META_CSV}")

    metadata = pd.read_csv(META_CSV)
    required = {
        "query_id",
        "decision_type",
        "candidate_rank",
        "file_path",
        "n_frames",
        REFEREE_COL,
    }
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError("Metadata columns are missing: " + ", ".join(missing))

    metadata["file_path"] = metadata["file_path"].apply(make_abs_path)
    metadata["label"] = (
        metadata["decision_type"].astype(str).str.lower().eq("incorrect")
    ).astype(int)
    decision_types = set(metadata["decision_type"].astype(str).str.lower())
    if not decision_types.issubset({"correct", "incorrect"}):
        raise ValueError("Only correct and incorrect decision types are accepted.")
    metadata["candidate_rank"] = pd.to_numeric(
        metadata["candidate_rank"], errors="coerce"
    )
    metadata["n_frames"] = pd.to_numeric(metadata["n_frames"], errors="coerce")
    return metadata


def _query_referees(metadata: pd.DataFrame) -> dict[object, object]:
    """One referee is confirmed for every matched query list."""
    if metadata[REFEREE_COL].isna().any():
        raise ValueError(f"Missing values were found in {REFEREE_COL}.")

    query_referees: dict[object, object] = {}
    for query_id, group in metadata.groupby("query_id", sort=False):
        referees = group[REFEREE_COL].unique().tolist()
        if len(referees) != 1:
            raise ValueError(
                f"Query {query_id} contains clips from more than one referee."
            )
        query_referees[query_id] = referees[0]
    return query_referees


def split_dataset(metadata: pd.DataFrame):
    """Matched queries are partitioned by referee."""
    query_referees = _query_referees(metadata)
    if len(query_referees) != EXPECTED_TOTAL_QUERIES:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_QUERIES} complete queries, "
            f"but {len(query_referees)} were found."
        )
    referee_counts = pd.Series(query_referees).value_counts()

    referee_order = sorted(referee_counts.index.tolist(), key=str)
    random = np.random.default_rng(SPLIT_SEED)
    random.shuffle(referee_order)

    if len(referee_order) <= N_TEST_REFEREES + 1:
        raise ValueError("Too few referees were found for the three partitions.")

    test_referees = set(referee_order[:N_TEST_REFEREES])
    remaining = referee_order[N_TEST_REFEREES:]

    validation_referees: Set[object] = set()
    validation_query_count = 0
    for referee in remaining:
        if validation_query_count >= MINIMUM_VALIDATION_QUERIES:
            break
        validation_referees.add(referee)
        validation_query_count += int(referee_counts.loc[referee])

    training_referees = set(remaining) - validation_referees
    if not training_referees or not validation_referees:
        raise ValueError("A training or validation partition could not be formed.")

    def queries_for(referees: Set[object]) -> Set[object]:
        return {
            query_id
            for query_id, referee in query_referees.items()
            if referee in referees
        }

    training_queries = queries_for(training_referees)
    validation_queries = queries_for(validation_referees)
    test_queries = queries_for(test_referees)

    def select(queries: Set[object]) -> pd.DataFrame:
        return metadata.loc[metadata["query_id"].isin(queries)].reset_index(drop=True)

    partitions = (
        select(training_queries),
        select(validation_queries),
        select(test_queries),
    )

    referee_sets = (training_referees, validation_referees, test_referees)
    if any(
        referee_sets[i] & referee_sets[j] for i in range(3) for j in range(i + 1, 3)
    ):
        raise RuntimeError("A referee was assigned to more than one partition.")

    query_sets = (training_queries, validation_queries, test_queries)
    if any(query_sets[i] & query_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("A query was assigned to more than one partition.")

    observed_counts = {
        "train": len(training_queries),
        "validation": len(validation_queries),
        "test": len(test_queries),
    }
    if observed_counts != EXPECTED_QUERY_COUNTS:
        raise ValueError(
            f"The referee split produced {observed_counts}, "
            f"but {EXPECTED_QUERY_COUNTS} was expected."
        )

    print(
        "[split] "
        f"train={len(training_queries)} queries, "
        f"validation={len(validation_queries)} queries, "
        f"test={len(test_queries)} queries",
        flush=True,
    )
    return (*partitions, *query_sets)


def _validate_feature_schema() -> None:
    """The stored feature order is checked against the model feature order."""
    if not SCHEMA_JSON.exists():
        raise FileNotFoundError(f"The feature schema was not found at {SCHEMA_JSON}")
    with SCHEMA_JSON.open("r", encoding="utf-8") as file:
        stored_columns = tuple(json.load(file)["ordered_columns"])
    if stored_columns != MODEL_FEATURE_COLUMNS:
        raise ValueError("The extracted feature order does not match config.py.")


def fit_scaler(metadata: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Feature scaling is fitted on valid training frames only."""
    if metadata.empty:
        raise ValueError("The training metadata is empty.")
    _validate_feature_schema()

    feature_count = len(MODEL_FEATURE_COLUMNS)
    sums = np.zeros(feature_count, dtype=np.float64)
    squared_sums = np.zeros(feature_count, dtype=np.float64)
    counts = np.zeros(feature_count, dtype=np.int64)

    for _, file_paths, lengths, _ in build_query_lists(metadata):
        shared_length = min(lengths)
        for file_path in file_paths:
            sequence = coerce_numeric(np.load(file_path, allow_pickle=False))
            if sequence.ndim != 2 or sequence.shape[1] != feature_count:
                raise ValueError(f"An invalid sequence shape was found in {file_path}.")

            sequence = sequence[-shared_length:, :]
            valid = np.isfinite(sequence)
            finite_values = np.where(valid, sequence, 0.0)
            sums += finite_values.sum(axis=0)
            squared_sums += np.square(finite_values).sum(axis=0)
            counts += valid.sum(axis=0)

    missing_features = [
        MODEL_FEATURE_COLUMNS[index] for index in np.flatnonzero(counts == 0)
    ]
    if missing_features:
        raise ValueError(
            "No valid training values were found for: " + ", ".join(missing_features)
        )

    means = sums / counts
    variances = (squared_sums / counts) - np.square(means)
    standard_deviations = np.sqrt(np.maximum(variances, MIN_STD**2))
    return {
        "keep_cols": np.arange(feature_count, dtype=int),
        "mean": means.astype(np.float32),
        "std": standard_deviations.astype(np.float32),
        "feature_dim": feature_count,
    }


def filter_valid_queries_listwise(
    metadata: pd.DataFrame,
    top_k: int = TOP_K_CORRECT,
) -> pd.DataFrame:
    """Only complete matched lists with sufficient clip length are retained."""
    expected_correct_ranks = list(range(1, top_k + 1))
    valid_queries: List[object] = []

    for query_id, group in metadata.groupby("query_id", sort=False):
        incorrect = group.loc[group["label"].eq(1)]
        correct = group.loc[group["label"].eq(0)].sort_values("candidate_rank")

        if len(incorrect) != 1:
            continue
        incorrect_rank = incorrect.iloc[0]["candidate_rank"]
        if not np.isfinite(incorrect_rank) or float(incorrect_rank) != 0.0:
            continue
        correct_ranks = correct["candidate_rank"].to_numpy(dtype=float)
        if not np.isfinite(correct_ranks).all():
            continue
        if correct_ranks.tolist() != expected_correct_ranks:
            continue
        if group["n_frames"].isna().any():
            continue
        if (group["n_frames"].astype(int) < MIN_CLIP_FRAMES).any():
            continue
        valid_queries.append(query_id)

    return metadata.loc[metadata["query_id"].isin(valid_queries)].reset_index(drop=True)


def build_query_lists(
    metadata: pd.DataFrame,
    top_k: int = TOP_K_CORRECT,
) -> List[Tuple[object, List[str], List[int], List[int]]]:
    """One ordered candidate list is built for every matched query."""
    items: List[Tuple[object, List[str], List[int], List[int]]] = []
    expected_ranks = list(range(1, top_k + 1))

    for query_id, group in metadata.groupby("query_id", sort=False):
        incorrect = group.loc[group["label"].eq(1)]
        correct = group.loc[group["label"].eq(0)].sort_values("candidate_rank")
        correct_ranks_array = correct["candidate_rank"].to_numpy(dtype=float)
        correct_ranks = (
            correct_ranks_array.tolist()
            if np.isfinite(correct_ranks_array).all()
            else []
        )

        if len(incorrect) != 1 or correct_ranks != expected_ranks:
            raise ValueError(f"Query {query_id} is not a complete matched list.")

        incorrect_row = incorrect.iloc[0]
        if not np.isfinite(incorrect_row["candidate_rank"]):
            raise ValueError(f"Query {query_id} has an invalid incorrect rank.")
        if float(incorrect_row["candidate_rank"]) != 0.0:
            raise ValueError(
                f"Query {query_id} does not place the incorrect clip first."
            )
        file_paths = [incorrect_row["file_path"], *correct["file_path"].tolist()]
        lengths = [
            min(int(length), TGT_LEN)
            for length in [incorrect_row["n_frames"], *correct["n_frames"].tolist()]
        ]
        items.append((query_id, file_paths, lengths, [0, *expected_ranks]))

    return items
