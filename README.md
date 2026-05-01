# QSeat

**Maintainer:** Marc Breiner Sørensen  
**License:** Apache-2.0

**QSeat** is an open-source, QUBO-based office seating optimizer. Given a set of 
employees, rooms, seats, attendance schedules, and pairwise affinities, QSeat computes 
an optimal weekly seating plan using quantum-inspired combinatorial optimization running 
on standard CPU hardware.

The problem is formally equivalent to a **Weighted Quadratic Assignment Problem (QAP)** (an NP-hard problem), and is solved 
via a QUBO formulation amenable to simulated annealing and tabu search samplers.

## Features

- Handles multiple rooms with arbitrary desk counts
- Supports fixed seat assignments (accessibility, management requirements, etc.)
- Accounts for daily attendance: employees absent all week are automatically unassigned
- Null-seat mechanism gracefully handles cases where employees exceed available seats
- Swappable solver backend (simulated annealing, tabu search)
- REST API via FastAPI for integration with frontends and data pipelines

## Model

A full mathematical derivation is available in [`docs/model.pdf`](docs/model.pdf). 
A brief summary follows.

### Inputs

| Symbol | Dimensions | Description |
|--------|------------|-------------|
| $P$ | $N \times N$ | Pairwise employee affinity matrix. $P_{nm}$ encodes how much employee $n$ needs to sit near employee $m$. In general asymmetric but symmetrized before solving. |
| $W$ | $N \times T$ | Attendance matrix. $W_{nt} = 1$ if employee $n$ is onsite on day $t$. $T = 5$ for a workweek. |
| $F$ | $N \times S$ | Fixed assignment matrix. $F_{ns} = 1$ if employee $n$ must be assigned to seat $s$. |
| $A$ | $(S+1) \times (S+1)$ | Seat adjacency matrix. $A_{kl} = 1$ if seats $k$ and $l$ are adjacent. Upper triangular, zero diagonal. Last row and column are zero (null seat). |

### Decision variable

$X \in \mathbb{B}^{N \times (S+1)}$ is the assignment matrix, where $X_{ns} = 1$ means 
employee $n$ is assigned to seat $s$ for the week. The $(S+1)$-th column is a **null 
seat**: employees assigned here receive no physical seat, allowing the problem to remain 
feasible when $N > S$. Each row of $X$ is a one-hot encoded vector (a standard basis 
vector), making $X$ a partial permutation matrix.

### Preprocessing

Before constructing the QUBO, three elimination steps reduce problem size:

1. **Absent employees** : any employee with $\sum_t W_{nt} = 0$ is immediately 
null-assigned and removed from the problem.
2. **Fixed assignments** : employees with $F_{ns} = 1$ are assigned directly via 
variable elimination, removing them and their seat from the free variables.
3. **Affinity modulation** : the raw affinity matrix is scaled by shared attendance days:

$$\tilde{P}_{nm} = P_{nm} \cdot \sum_{t=1}^{T} W_{nt} W_{mt}$$

### Objective

$$\min_X \; -\sum_{n,m=1}^{N} \sum_{s,s'=1}^{S} \tilde{P}_{nm} A_{ss'} X_{ns} X_{ms'}$$

This rewards placing high-affinity employee pairs at adjacent seats.

### Constraints

One-hot structure on $X$ is enforced via two penalty terms:

$$P_{\text{row}} = \sum_{n=1}^{N} \left(1 - \sum_{s=1}^{S+1} X_{ns}\right)^2$$

$$P_{\text{col}} = \sum_{s=1}^{S} \sum_{n < m} X_{ns} X_{ms}$$

Fixed assignments are penalized via:

$$P_{\text{fixed}} = \sum_{n,s} F_{ns}(1 - X_{ns})^2 + \sum_{n} \mathbf{1}[\exists s : F_{ns} = 1] \sum_{s' \neq s} X_{ns'}^2$$

though in practice $P_{\text{fixed}}$ is handled via variable elimination rather than 
as an explicit penalty term.

### Full QUBO

$$\min_x \; x^\top Q_{\text{QUBO}}\, x, \quad Q_{\text{QUBO}} = -\sum_{nm,ss'} \tilde{P}_{nm} A_{ss'} \cdot e_{ns} e_{ms'}^\top + \lambda_1 P_{\text{row}} + \lambda_2 P_{\text{col}}$$

Penalty weights should satisfy $\lambda_1, \lambda_2 > \max_{nm} |\tilde{P}_{nm}| \cdot S$ 
to ensure no constraint violation is energetically favourable.

## Installation

```bash
pip install qseat
```

For development:

```bash
git clone https://github.com/yourname/qseat
cd qseat
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

_Coming soon._

## API

_Coming soon._

## Contributing

Contributions are welcome. Please open an issue before submitting a pull request.

## License

Apache-2.0. See [LICENSE](LICENSE) for details.
