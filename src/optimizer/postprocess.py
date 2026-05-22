"""
Postprocessing for QuESO.

Provides domain-specific interpretation of raw dimod SampleSet outputs:

    is_valid()        -- checks row and column constraints on the best sample
    decode()          -- translates flat sample to full employee -> seat mapping
    objective_value() -- computes raw affinity objective excluding penalties
    null_assigned()   -- returns employee indices assigned to the null seat
"""

import dimod
import numpy as np

from util import mu, get_X_from_x
from .preprocess import PreprocessingResult


def _decode_active(
        sample: dict,
        active_map: dict[int, int],
        S: int,
) -> dict[int, int]:
    """
    Decode assignments for active employees from the flat sample.

    Parameters
    ----------
    sample : dict
        Best sample from the SampleSet.
    active_map : dict[int, int]
        Mapping of reduced employee index -> original employee index.
    S : int
        Number of real seats. Null seat is at index S.

    Returns
    -------
    dict[int, int]
        Partial assignment: original employee index -> seat index.
    """
    assignment = {}
    for idxs_ns_reduced_matrix, idxs_ns_original_matrix in active_map.items():
        row = np.array([sample[mu(idxs_ns_reduced_matrix, s, S)] for s in range(S + 1)])
        assignment[idxs_ns_original_matrix] = int(np.argmax(row)) if row.sum() == 1 else S
    return assignment


def _decode_fixed(pre_fixed: dict[int, int]) -> dict[int, int]:
    """
    Decode assignments for fixed-seat employees.

    Parameters
    ----------
    pre_fixed : dict[int, int]
        Mapping of original employee index -> seat index from preprocessing.

    Returns
    -------
    dict[int, int]
        Partial assignment: original employee index -> seat index.
    """
    return dict(pre_fixed)


def _decode_absent(no_attendance: list[int], S: int) -> dict[int, int]:
    """
    Decode assignments for absent employees (all null-assigned).

    Parameters
    ----------
    no_attendance : list[int]
        Original indices of employees with no attendance this week.
    S : int
        Number of real seats. Null seat is at index S.

    Returns
    -------
    dict[int, int]
        Partial assignment: original employee index -> S (null seat).
    """
    return {n: S for n in no_attendance}


def decode(
    sample_set: dimod.SampleSet,
    result: PreprocessingResult,
) -> dict[int, int]:
    """
    Translate the best flat sample into a full assignment mapping.

    Combines the solver output for active employees with the pre-eliminated
    employees from the preprocessing step to produce a complete mapping
    from original employee index to seat index.

    Seat index S+1 (using original S, i.e. result.A.shape[0] - 1) denotes
    null assignment (no seat this week).

    Parameters
    ----------
    sample_set : dimod.SampleSet
        Raw output from the solver.
    result : PreprocessingResult
        Output of the preprocessing pipeline, used for index mapping.

    Returns
    -------
    dict[int, int]
        Mapping of original employee index -> seat index (0-indexed).
        Null-assigned employees map to S (the null seat index).
    """
    S = result.A.shape[0] - 1
    sample = sample_set.first.sample

    index_map = result.employee_index_map

    return {
        **_decode_active(sample, index_map["active"], S),
        **_decode_fixed(index_map["pre_fixed"]),
        **_decode_absent(index_map["no_attendance"], S),
    }


def objective_value(
    sample_set: dimod.SampleSet,
    result: PreprocessingResult,
) -> float:
    """
    Compute the raw affinity objective for the best sample.

    Evaluates the objective without penalty terms:
        -sum_{n,m,s,s'} P_tilde_nm * A_ss' * X_ns * X_ms'

    This gives a penalty-independent measure of solution quality,
    useful for comparing solutions across different penalty scalings.

    Parameters
    ----------
    sample_set : dimod.SampleSet
        Raw output from the solver.
    result : PreprocessingResult
        Contains P_tilde and A needed to evaluate the objective.

    Returns
    -------
    float
        Raw affinity objective value (negative = better).
    """
    S = result.A.shape[0] - 1
    N_active = result.n_active
    sample = sample_set.first.sample

    # Reconstruct X matrix for active employees
    x = np.array([sample[i] for i in sorted(sample.keys())], dtype=float)
    X = get_X_from_x(x, N_active, S)

    # Objective: -tr(P_tilde @ X @ A @ X.T)
    # Equivalent to -sum_{n,m,s,s'} P_tilde_nm * A_ss' * X_ns * X_ms'
    obj = -np.einsum(
        "nm,ns,ss',ms'->",
        result.P_tilde,
        X[:, :-1],
        result.A[:-1, :-1].astype(float),
        X[:, :-1],
    )
    return float(obj)


def null_assigned(decoded: dict[int, int], S: int) -> list[int]:
    """
    Return the list of original employee indices assigned to the null seat.

    Parameters
    ----------
    decoded : dict[int, int]
        Output of decode(), mapping original employee index -> seat index.
    S : int
        Total number of real seats. Null seat is at index S.

    Returns
    -------
    list[int]
        Sorted list of original employee indices with no seat this week.
    """
    return sorted(n for n, seat in decoded.items() if seat == S)