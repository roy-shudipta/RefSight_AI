"""Multivariate dynamic time warping is calculated with Numba."""

from __future__ import annotations

import math

import numba as nb
import numpy as np


_INFINITY = 1e308


def _as_float64_contiguous(sequence: np.ndarray) -> np.ndarray:
    """A contiguous float64 sequence is returned."""
    return np.ascontiguousarray(sequence, dtype=np.float64)


@nb.njit(cache=True, nogil=True, fastmath=True)
def _lb_keogh(candidate: np.ndarray, upper: np.ndarray, lower: np.ndarray) -> float:
    """The multivariate LB_Keogh lower bound is calculated."""
    frame_count, feature_count = candidate.shape
    total = 0.0
    for frame in range(frame_count):
        squared_distance = 0.0
        for feature in range(feature_count):
            value = candidate[frame, feature]
            if value > upper[frame, feature]:
                difference = value - upper[frame, feature]
                squared_distance += difference * difference
            elif value < lower[frame, feature]:
                difference = lower[frame, feature] - value
                squared_distance += difference * difference
        total += math.sqrt(squared_distance)
    return total


@nb.njit(cache=True, nogil=True, fastmath=True)
def _envelope(sequence: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Upper and lower LB_Keogh envelopes are calculated."""
    frame_count, feature_count = sequence.shape
    upper = np.empty((frame_count, feature_count), dtype=np.float64)
    lower = np.empty((frame_count, feature_count), dtype=np.float64)

    for frame in range(frame_count):
        start = max(0, frame - width)
        end = min(frame_count - 1, frame + width)
        for feature in range(feature_count):
            maximum = -_INFINITY
            minimum = _INFINITY
            for neighbour in range(start, end + 1):
                value = sequence[neighbour, feature]
                maximum = max(maximum, value)
                minimum = min(minimum, value)
            upper[frame, feature] = maximum
            lower[frame, feature] = minimum
    return upper, lower


@nb.njit(cache=True, nogil=True, fastmath=True)
def _dtw_distance_kernel(
    query: np.ndarray,
    candidate: np.ndarray,
    width: int,
    best_so_far: float,
) -> float:
    """A band-constrained DTW distance is calculated with early stopping."""
    query_frames, feature_count = query.shape
    candidate_frames = candidate.shape[0]
    previous = np.full(candidate_frames + 1, _INFINITY, dtype=np.float64)
    current = np.full(candidate_frames + 1, _INFINITY, dtype=np.float64)
    previous[0] = 0.0

    for query_frame in range(1, query_frames + 1):
        candidate_start = max(1, query_frame - width)
        candidate_end = min(candidate_frames, query_frame + width)
        current[:] = _INFINITY
        row_minimum = _INFINITY

        for candidate_frame in range(candidate_start, candidate_end + 1):
            squared_distance = 0.0
            for feature in range(feature_count):
                difference = (
                    query[query_frame - 1, feature]
                    - candidate[candidate_frame - 1, feature]
                )
                squared_distance += difference * difference
            local_cost = math.sqrt(squared_distance)
            previous_row = previous[candidate_frame]
            current_row = current[candidate_frame - 1]
            diagonal = previous[candidate_frame - 1]
            current[candidate_frame] = local_cost + min(
                previous_row,
                current_row,
                diagonal,
            )
            row_minimum = min(row_minimum, current[candidate_frame])

        if row_minimum > best_so_far:
            return _INFINITY
        previous, current = current, previous
    return previous[candidate_frames]


def dtw_distance(
    query: np.ndarray,
    candidate: np.ndarray,
    band_frac: float,
    best_so_far: float = _INFINITY,
) -> float:
    """The exact multivariate DTW distance is returned."""
    query = _as_float64_contiguous(query)
    candidate = _as_float64_contiguous(candidate)
    width = int(math.ceil(band_frac * max(len(query), len(candidate))))
    return float(_dtw_distance_kernel(query, candidate, width, best_so_far))


def _query_envelope(
    query: np.ndarray,
    band_frac: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The query envelope used by LB_Keogh is returned."""
    query = _as_float64_contiguous(query)
    width = int(math.ceil(band_frac * len(query)))
    return _envelope(query, width)


def two_phase_retrieve(
    q_seq: np.ndarray,
    cand_seqs: list[np.ndarray],
    k: int,
    band_frac: float,
    initial_best: float,
    seed: int = 12345,
) -> list[tuple[int, float]]:
    """Cross-match candidates are retrieved with LB_Keogh pruning."""
    if not cand_seqs or k <= 0:
        return []

    query = _as_float64_contiguous(q_seq)
    query_frames = len(query)
    upper, lower = _query_envelope(query, band_frac)
    random_generator = np.random.default_rng(seed)
    candidate_indices = np.arange(len(cand_seqs), dtype=int)
    seed_indices = random_generator.choice(
        candidate_indices,
        size=min(1000, len(cand_seqs)),
        replace=False,
    )

    best: list[tuple[int, float]] = []
    kth_distance = float(initial_best)

    # Seeded exact DTW distances are used to set the pruning threshold.
    for candidate_index in seed_indices:
        candidate = cand_seqs[int(candidate_index)]
        if len(candidate) > query_frames:
            candidate = candidate[-query_frames:]
        distance = dtw_distance(
            query,
            candidate,
            band_frac=band_frac,
            best_so_far=kth_distance,
        )
        best.append((int(candidate_index), float(distance)))

    best.sort(key=lambda item: item[1])
    best = best[:k]
    if best:
        kth_distance = min(kth_distance, best[-1][1])

    # LB_Keogh is used to remove candidates that cannot enter the top-k list.
    seed_set = set(int(index) for index in seed_indices.tolist())
    survivors: list[int] = []
    for candidate_index, candidate in enumerate(cand_seqs):
        if candidate_index in seed_set:
            continue
        if len(candidate) > query_frames:
            candidate = candidate[-query_frames:]
        candidate = _as_float64_contiguous(candidate)
        candidate_frames = len(candidate)
        candidate_upper = upper[-candidate_frames:, :]
        candidate_lower = lower[-candidate_frames:, :]
        lower_bound = float(_lb_keogh(candidate, candidate_upper, candidate_lower))
        if lower_bound < kth_distance:
            survivors.append(candidate_index)

    # Exact DTW is calculated for every surviving candidate.
    for candidate_index in survivors:
        candidate = cand_seqs[candidate_index]
        if len(candidate) > query_frames:
            candidate = candidate[-query_frames:]
        distance = dtw_distance(
            query,
            candidate,
            band_frac=band_frac,
            best_so_far=kth_distance,
        )
        best.append((candidate_index, float(distance)))
        best.sort(key=lambda item: item[1])
        best = best[:k]
        kth_distance = min(kth_distance, best[-1][1])
    return best
