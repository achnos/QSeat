"""
QuESO FastAPI application.

Exposes the QUBO-based seating optimizer as an HTTP API.

Endpoints:
    GET  /         -- package info
    GET  /health   -- health check
    POST /optimize -- run the seating optimizer
"""

import importlib.metadata

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, model_validator

from .optimizer.preprocess import preprocess
from .optimizer.qubo import build_qubo
from .optimizer.postprocess import decode, null_assigned, objective_value
from util.util import is_valid_solution
from .optimizer.solver import solve_with_retry

app = FastAPI(
    title="QuESO",
    description="QuESO (Quantum Enhanced Seat Optimizer) is a QUBO-based office seating optimizer",
    version=importlib.metadata.version("queso"),
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class SeatingRequest(BaseModel):
    P: list[list[float | int]]  # (N, N)  affinity matrix
    W: list[list[int]]          # (N, 5)  attendance matrix
    F: list[list[int]]          # (N, S)  fixed assignment matrix
    A: list[list[int]]          # (S+1, S+1) seat adjacency matrix
    lambda_1: float | None = None
    lambda_2: float | None = None
    penalty_multiplier: float = 2.0
    max_retries: int = 3

    @model_validator(mode="after")
    def check_dimensions(self) -> "SeatingRequest":
        N = len(self.P)
        S = len(self.A) - 1
        if any(len(row) != N for row in self.P):
            raise ValueError("P must be square (N x N)")
        if any(len(row) != 5 for row in self.W):
            raise ValueError("W must have 5 columns (one per workday)")
        if len(self.W) != N:
            raise ValueError("W must have N rows")
        if len(self.F) != N:
            raise ValueError("F must have N rows")
        if any(len(row) != S + 1 - 1 for row in self.F):
            raise ValueError("F must have S columns (excluding null seat)")
        if any(len(row) != S + 1 for row in self.A):
            raise ValueError("A must be square (S+1 x S+1)")
        if self.penalty_multiplier <= 0:
            raise ValueError("penalty_multiplier must be positive")
        if self.max_retries < 1:
            raise ValueError("max_retries must be at least 1")

        return self


class SeatingResponse(BaseModel):
    assignment_map: dict[int, int]
    null_assigned: list[int]
    objective_value: float
    valid_solution: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root() -> dict:
    return {
        "name": "QuESO",
        "version": importlib.metadata.version("queso"),
        "description": "QuESO (Quantum Enhanced Seat Optimizer) is a QUBO-based office seating optimizer",
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/optimize", response_model=SeatingResponse)
def optimize(request: SeatingRequest) -> SeatingResponse:
    # Convert to numpy
    P = np.array(request.P, dtype=float)
    W = np.array(request.W, dtype=int)
    F = np.array(request.F, dtype=int)
    A = np.array(request.A, dtype=int)

    S = A.shape[0] - 1

    # Preprocessing
    result = preprocess(P, W, F, A)

    # QUBO construction
    bqm = build_qubo(
        result,
        lambda_1=request.lambda_1,
        lambda_2=request.lambda_2,
        penalty_multiplier=request.penalty_multiplier,
    )

    sample_set, valid = solve_with_retry(
        bqm,
        is_valid=lambda ss: is_valid_solution(ss, result.n_active, S),
        max_retries=request.max_retries,
    )

    # Postprocess
    assignment = decode(sample_set, result)
    unassigned = null_assigned(assignment, S)
    obj = objective_value(sample_set, result)

    return SeatingResponse(
        assignment_map=assignment,
        null_assigned=unassigned,
        objective_value=obj,
        valid_solution=valid,
    )