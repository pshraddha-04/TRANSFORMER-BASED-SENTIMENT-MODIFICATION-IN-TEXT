import logging
from collections import Counter
from collections import deque
from threading import Lock

from app.config import DRIFT_DOMINANCE_THRESHOLD, DRIFT_LOG_EVERY_N, DRIFT_WINDOW_SIZE


logger = logging.getLogger("app.monitoring")


class PredictionMonitor:
    def __init__(
        self,
        log_every_n: int = DRIFT_LOG_EVERY_N,
        dominance_threshold: float = DRIFT_DOMINANCE_THRESHOLD,
        window_size: int = DRIFT_WINDOW_SIZE,
    ) -> None:
        self.log_every_n = max(1, int(log_every_n))
        self.dominance_threshold = float(dominance_threshold)
        self.window_size = max(1, int(window_size))
        self._counts: Counter[str] = Counter()
        self._window_counts: Counter[str] = Counter()
        self._window: deque[str] = deque(maxlen=self.window_size)
        self._total = 0
        self._lock = Lock()

    def _window_distribution(self) -> dict[str, float]:
        window_total = len(self._window)
        if window_total == 0:
            return {}
        return {label: round(count / window_total, 4) for label, count in self._window_counts.items()}

    def _append_to_window(self, label: str) -> None:
        if len(self._window) == self._window.maxlen:
            dropped = self._window.popleft()
            self._window_counts[dropped] -= 1
            if self._window_counts[dropped] <= 0:
                del self._window_counts[dropped]
        self._window.append(label)
        self._window_counts[label] += 1

    def record_prediction(self, emotion: str) -> None:
        label = (emotion or "unknown").strip().lower() or "unknown"
        with self._lock:
            self._counts[label] += 1
            self._total += 1
            self._append_to_window(label)
            should_log = self._total % self.log_every_n == 0
            total = self._total
            distribution = (
                {key: round(value / total, 4) for key, value in self._counts.items()}
                if total
                else {}
            )
            window_distribution = self._window_distribution()
            window_total = len(self._window)

        if not should_log:
            return

        if window_distribution:
            dominant_label, dominant_ratio = max(window_distribution.items(), key=lambda item: item[1])
            if dominant_ratio >= self.dominance_threshold:
                logger.warning(
                    "Prediction distribution drift signal: label=%s ratio=%.3f threshold=%.3f window=%s total=%s",
                    dominant_label,
                    dominant_ratio,
                    self.dominance_threshold,
                    window_total,
                    total,
                )

        logger.info(
            "Prediction distribution snapshot: total=%s window=%s lifetime=%s recent=%s",
            total,
            window_total,
            distribution,
            window_distribution,
        )

    def snapshot(self) -> dict:
        with self._lock:
            if self._total == 0:
                return {
                    "total": 0,
                    "counts": {},
                    "distribution": {},
                    "window_total": 0,
                    "window_counts": {},
                    "window_distribution": {},
                    "window_size": self.window_size,
                }

            distribution = {label: round(count / self._total, 4) for label, count in self._counts.items()}
            window_total = len(self._window)
            window_distribution = self._window_distribution()
            return {
                "total": self._total,
                "counts": dict(self._counts),
                "distribution": distribution,
                "window_total": window_total,
                "window_counts": dict(self._window_counts),
                "window_distribution": window_distribution,
                "window_size": self.window_size,
            }


prediction_monitor = PredictionMonitor()

