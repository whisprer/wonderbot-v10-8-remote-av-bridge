from __future__ import annotations

import math
from typing import Iterable, List


class ResonanceField:
    def __init__(self, sigma: float = 0.5, tau: float = 14.134725, alpha: float = 1.2, prime_count: int = 32) -> None:
        self.sigma = sigma
        self.tau = tau
        self.alpha = alpha
        self.primes = _small_primes(prime_count)

    def score_signature(self, signature: str, tick: int) -> float:
        core = int(signature[:16], 16)
        total = 0.0
        for prime in self.primes:
            phase = ((core % (prime * 997)) / max(1, prime * 997)) * 2.0 * math.pi
            weight = (prime ** (-self.sigma)) * math.cos(self.tau * math.log(prime) + phase + tick * 0.05)
            total += weight
        scale = math.sqrt(max(1, len(self.primes)))
        return math.tanh((self.alpha * total) / scale)

    def score_many(self, signatures: Iterable[str], tick: int) -> float:
        values = [self.score_signature(sig, tick) for sig in signatures]
        if not values:
            return 0.0
        return sum(values) / len(values)

    def should_react(self, salience: float, reaction_threshold: float, explicit: bool) -> bool:
        if explicit:
            return True
        return salience >= reaction_threshold


def _small_primes(n: int) -> List[int]:
    primes: List[int] = []
    candidate = 2
    while len(primes) < max(1, n):
        is_prime = True
        limit = int(math.sqrt(candidate)) + 1
        for p in range(2, limit):
            if candidate % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1
    return primes
