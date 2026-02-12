# utils/meters.py
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class AvgMeter:
    total: dict = field(default_factory=dict)
    count: dict = field(default_factory=dict)

    def update(self, loss: float, n: int, parts: dict):
        self._add("loss", loss, n)
        for k, v in parts.items():
            self._add(k, v, n)

    def _add(self, k: str, v: float, n: int):
        self.total[k] = self.total.get(k, 0.0) + float(v) * n
        self.count[k] = self.count.get(k, 0) + n

    def avg(self, k: str):
        return self.total.get(k, 0.0) / max(1, self.count.get(k, 1))
