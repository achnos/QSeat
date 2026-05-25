from typing import Any

import dimod
import numpy as np
from dimod import SampleSet


def symmetrize_matrix(P_tilde: np.ndarray) -> np.ndarray:
    return (P_tilde + P_tilde.T) / 2.0


def mu(n: int, s: int, S: int) -> int:
    """Flattening map"""
    return n * (S + 1) + s


def is_valid_solution(
    sample_set: dimod.SampleSet,
    n_active: int,
    S: int,
) -> bool:
    """
    Check whether the sample satisfies all constraints.

    Constraints checked:
    - Row constraint: each active employee is assigned exactly one seat
      (including the null seat at index S).
    - Column constraint: no two active employees share the same real seat
      (seats 0..S-1).

    Parameters
    ----------
    sample_set : dimod.SampleSet
        Raw output from the solver.
    n_active : int
        Number of active employees (rows in the reduced X matrix).
    S : int
        Number of real seats (excluding null seat).

    Returns
    -------
    bool
        True if the sample satisfies all constraints.
    """
    x = get_best_sample_as_ndarray(sample_set)
    X = get_X_from_x(x, n_active, S)

    return is_each_employee_assigned_exactly_once(X) and is_each_seat_occupied_by_at_most_one_employee(X, S)


def get_X_from_x(x: np.ndarray, n_active: int, S: int) -> np.ndarray:
    return x.reshape(n_active, S + 1)


def is_each_employee_assigned_exactly_once(X: np.ndarray) -> bool:
    return np.all(X.sum(axis=1) == 1)


def is_each_seat_occupied_by_at_most_one_employee(X: np.ndarray, S: int) -> bool:
    real_seats = get_real_seat_assignments(X, S)
    return np.all(real_seats.sum(axis=0) <= 1)


def get_real_seat_assignments(X: np.ndarray,  S: int) -> np.ndarray:
    real_seats = X[:, :S]
    return real_seats


def get_best_sample(sample_set: SampleSet) -> dimod.sampleset.SampleView:
    return sample_set.first.sample


def get_best_sample_as_ndarray(sample_set: SampleSet) -> np.ndarray:
    return sample_set.record.sample[0]
