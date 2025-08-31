from __future__ import annotations
import numpy as np
from strategy import Strategy


class RPSStrategy(Strategy):
    PAYOFF_TABLE = np.array([[0, -1,  1],
                             [1,  0, -1],
                             [-1, 1,  0]], dtype=float)

    def __init__(self, utility_fn, mode="expected_utility", origin=None, mapping=None):
        self.utility_fn = utility_fn
        self.mode = mode

        if origin is None:
            origin = {w: 1.0 / 10.0 for w in range(1, 11)}
        if mapping is None:
            mapping = {
                0: [1, 2, 3, 4, 5],
                1: [6, 7, 8, 9],
                2: [10],
            }

        self.opponent_dist = self.compute_distribution(origin, mapping)
        self._p = np.array(self.opponent_dist, dtype=float)

    def decision(self, rng: np.random.Generator) -> int:
        if self.mode == "expected_payoff":
            values = self.expected_payoff(self._p, self.PAYOFF_TABLE)
        else:
            values = self.expected_utility(self._p, self.PAYOFF_TABLE, self.utility_fn)

        best = np.flatnonzero(values == values.max())
        return int(rng.choice(best))
