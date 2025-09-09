"""
Ant Colony Optimization (ACO) for the Traveling Salesman Problem (TSP)

Pure-Python, single-file reference implementation with clear structure and docs.
- Symmetric TSP (distance(i, j) == distance(j, i))
- Euclidean distance between 2D city coordinates
- Standard ACS-like probabilistic construction (no local search) with global pheromone update

Usage (run this file directly):
    python aco_tsp.py

You can also import ACOTSP into your own scripts.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Sequence, Optional

City = Tuple[float, float]


def euclidean_distance(a: City, b: City) -> float:
    """Euclidean distance between two 2D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def build_distance_matrix(cities: Sequence[City]) -> List[List[float]]:
    """Pre-compute a symmetric distance matrix for faster evaluation."""
    n = len(cities)
    d = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dist = euclidean_distance(cities[i], cities[j])
            d[i][j] = d[j][i] = dist
    return d


def tour_length(tour: Sequence[int], dist: List[List[float]]) -> float:
    """Compute the closed-tour length (wraps back to start)."""
    n = len(tour)
    total = 0.0
    for i in range(n):
        a = tour[i]
        b = tour[(i + 1) % n]
        total += dist[a][b]
    return total


@dataclass
class ACOParams:
    n_ants: int = 20
    n_iterations: int = 200
    alpha: float = 1.0            # pheromone importance
    beta: float = 5.0             # heuristic (1/distance) importance
    rho: float = 0.5              # evaporation rate in [0,1]
    q: float = 1.0                # pheromone deposit factor (Q)
    initial_pheromone: float = 1.0
    elite_weight: float = 0.0     # extra deposit on global-best edges each iter (0 = off)
    seed: Optional[int] = None
    patience: Optional[int] = None  # stop if no improvement for this many iterations


class ACOTSP:
    """Ant Colony Optimization for symmetric Euclidean TSP.

    Attributes set from params:
        - n_ants, n_iterations, alpha, beta, rho, q, initial_pheromone, elite_weight, seed, patience

    Public API:
        - solve(cities) -> (best_tour, best_length, history)
    """

    def __init__(self, params: ACOParams):
        self.params = params
        if self.params.seed is not None:
            random.seed(self.params.seed)

        # Will be set during solve()
        self._global_best_tour: Optional[List[int]] = None
        self._global_best_length: float = float("inf")

    # ---------------------- Core helpers ----------------------
    def _init_pheromone(self, n: int) -> List[List[float]]:
        tau0 = self.params.initial_pheromone
        return [[tau0 if i != j else 0.0 for j in range(n)] for i in range(n)]

    def _build_eta(self, dist: List[List[float]]) -> List[List[float]]:
        """Heuristic visibility matrix: eta[i][j] = 1 / distance(i,j)."""
        n = len(dist)
        eps = 1e-12
        eta = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    eta[i][j] = 1.0 / (dist[i][j] + eps)
        return eta

    def _select_next_city(
        self,
        current: int,
        unvisited: List[int],
        pheromone: List[List[float]],
        eta: List[List[float]],
    ) -> int:
        """Roulette-wheel selection using (tau^alpha * eta^beta)."""
        alpha, beta = self.params.alpha, self.params.beta

        weights = []
        denom = 0.0
        for j in unvisited:
            val = (pheromone[current][j] ** alpha) * (eta[current][j] ** beta)
            weights.append((j, val))
            denom += val

        if denom <= 0.0:  # fallback to uniform if everything underflows
            return random.choice(unvisited)

        r = random.random()
        acc = 0.0
        for city, val in weights:
            acc += val / denom
            if r <= acc:
                return city
        return weights[-1][0]  # floating-point safety net

    def _construct_tour(
        self, pheromone: List[List[float]], eta: List[List[float]]
    ) -> List[int]:
        n = len(pheromone)
        start = random.randrange(n)
        tour = [start]
        unvisited = list(range(n))
        unvisited.remove(start)
        current = start

        while unvisited:
            nxt = self._select_next_city(current, unvisited, pheromone, eta)
            tour.append(nxt)
            unvisited.remove(nxt)
            current = nxt
        return tour

    def _evaporate(self, pheromone: List[List[float]]):
        n = len(pheromone)
        decay = 1.0 - self.params.rho
        for i in range(n):
            row = pheromone[i]
            for j in range(n):
                if i != j:
                    row[j] *= decay
                    if row[j] < 1e-12:  # prevent numerical collapse
                        row[j] = 1e-12

    def _deposit(self, pheromone: List[List[float]], tour: Sequence[int], amount: float):
        n = len(tour)
        for i in range(n):
            a = tour[i]
            b = tour[(i + 1) % n]
            pheromone[a][b] += amount
            pheromone[b][a] += amount  # symmetric

    def _update_pheromone_global(
        self,
        pheromone: List[List[float]],
        ants_tours: List[List[int]],
        ants_lengths: List[float],
    ):
        self._evaporate(pheromone)

        # Deposit from each ant proportional to 1 / L_k (classic AS)
        for tour, length in zip(ants_tours, ants_lengths):
            self._deposit(pheromone, tour, self.params.q / length)

        # Optional: extra elite deposit on the global best tour
        if self.params.elite_weight > 0.0 and self._global_best_tour is not None:
            self._deposit(
                pheromone,
                self._global_best_tour,
                self.params.elite_weight * (self.params.q / self._global_best_length),
            )

    # ---------------------- Public solve() ----------------------
    def solve(self, cities: Sequence[City]):
        """Run ACO on the provided city coordinates.

        Returns:
            best_tour (List[int]): indices of cities in visiting order
            best_length (float): total length of the closed tour
            history (List[float]): best length after each iteration
        """
        dist = build_distance_matrix(cities)
        n = len(cities)
        eta = self._build_eta(dist)
        pheromone = self._init_pheromone(n)

        history: List[float] = []
        no_improve_counter = 0

        for it in range(self.params.n_iterations):
            ants_tours: List[List[int]] = []
            ants_lengths: List[float] = []

            # --- Construct solutions ---
            for _ in range(self.params.n_ants):
                tour = self._construct_tour(pheromone, eta)
                length = tour_length(tour, dist)
                ants_tours.append(tour)
                ants_lengths.append(length)

                if length < self._global_best_length:
                    self._global_best_length = length
                    self._global_best_tour = tour.copy()
                    no_improve_counter = 0

            # --- Update pheromone globally ---
            self._update_pheromone_global(pheromone, ants_tours, ants_lengths)

            # --- Track progress ---
            history.append(self._global_best_length)
            no_improve_counter += 1

            # Early stop on patience
            if self.params.patience is not None and no_improve_counter >= self.params.patience:
                break

        # Return the best found tour and its length
        return self._global_best_tour, self._global_best_length, history


# ---------------------- Demo / quick test ----------------------
if __name__ == "__main__":
    # Example: 20 random cities in a 100x100 square
    random.seed(42)
    N_CITIES = 20
    cities: List[City] = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(N_CITIES)]

    params = ACOParams(
        n_ants=30,
        n_iterations=300,
        alpha=1.0,
        beta=5.0,
        rho=0.5,
        q=1.0,
        initial_pheromone=1.0,
        elite_weight=2.0,  # try 0.0 to disable
        seed=123,
        patience=50,       # stop if no improvement for 50 iters
    )

    solver = ACOTSP(params)
    best_tour, best_len, history = solver.solve(cities)

    print("Best length:", round(best_len, 3))
    print("Best tour (city indices):", best_tour)
    print("First 10 cities with coords:")
    for i, c in list(enumerate(cities))[:10]:
        print(f"  {i}: {c}")

    # Optional: simple ASCII summary of improvement
    try:
        import statistics as _stats  # only used here; safe if present
        if history:
            improv = history[0] - history[-1]
            print(f"Total improvement: {improv:.3f} over {len(history)} iters")
            if len(history) >= 5:
                last5 = history[-5:]
                print("Last 5 best lengths:", ", ".join(f"{x:.2f}" for x in last5))
    except Exception:
        pass
