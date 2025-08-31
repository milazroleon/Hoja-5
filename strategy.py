from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod


class Strategy(ABC):
    def compute_distribution(self, origin: dict, mapping: dict) -> np.ndarray:
        total_prob = float(sum(origin.values()))
        if total_prob <= 0.0:
            raise ValueError("Sum of origin probabilities must be positive.")
        origin = {w: float(p) / total_prob for w, p in origin.items()}

        dist = np.zeros(3, dtype=float)
        for k, ws in mapping.items():
            if k not in (0, 1, 2):
                continue
            prob = 0.0
            for w in ws:
                prob += origin.get(w, 0.0)
            dist[k] = prob

        s = float(dist.sum())
        if s <= 0.0:
            raise ValueError("Induced distribution is degenerate (sum <= 0).")
        return dist / s

    def expected_payoff(self, opponent_dist, payoff_table) -> np.ndarray:
        if isinstance(opponent_dist, dict):
            p = np.array([opponent_dist.get(0, 0.0),
                          opponent_dist.get(1, 0.0),
                          opponent_dist.get(2, 0.0)], dtype=float)
        else:
            p = np.array(opponent_dist, dtype=float)
            if p.shape != (3,):
                raise ValueError("opponent_dist array must have shape (3,)")
        s = float(p.sum())
        if s <= 0.0:
            raise ValueError("Opponent distribution must sum to a positive value.")
        p = p / s

        A = np.array(payoff_table, dtype=float)
        if A.shape != (3, 3):
            raise ValueError("payoff_table must have shape (3,3)")
        return A @ p

    def expected_utility(self, opponent_dist, payoff_table, utility_fn) -> np.ndarray:
        if isinstance(opponent_dist, dict):
            p = np.array([opponent_dist.get(0, 0.0),
                          opponent_dist.get(1, 0.0),
                          opponent_dist.get(2, 0.0)], dtype=float)
        else:
            p = np.array(opponent_dist, dtype=float)
            if p.shape != (3,):
                raise ValueError("opponent_dist array must have shape (3,)")
        s = float(p.sum())
        if s <= 0.0:
            raise ValueError("Opponent distribution must sum to a positive value.")
        p = p / s

        A = np.array(payoff_table, dtype=float)
        if A.shape != (3, 3):
            raise ValueError("payoff_table must have shape (3,3)")
        util_A = np.vectorize(utility_fn, otypes=[float])(A)
        return util_A @ p

    @abstractmethod
    def decision(self, rng: np.random.Generator) -> int:
        raise NotImplementedError("Subclasses must implement this method.")
