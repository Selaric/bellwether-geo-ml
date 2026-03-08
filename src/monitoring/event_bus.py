"""
Observer Pattern — Decoupled event bus for pipeline monitoring and alerting.
Subscribers register for specific event types and are notified automatically.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    INGESTION_STARTED = "ingestion.started"
    INGESTION_COMPLETE = "ingestion.complete"
    INGESTION_FAILED = "ingestion.failed"
    PREDICTION_MADE = "prediction.made"
    HIGH_RISK_DETECTED = "risk.high_detected"
    MODEL_DRIFT_DETECTED = "model.drift_detected"
    PIPELINE_ERROR = "pipeline.error"


@dataclass
class Event:
    type: EventType
    payload: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = "unknown"


class EventHandler(ABC):
    """Base observer — all handlers implement this interface."""

    @abstractmethod
    def handle(self, event: Event) -> None:
        ...

    def __call__(self, event: Event) -> None:
        self.handle(event)


class EventBus:
    """
    Central pub/sub event bus.
    Handlers subscribe per EventType; fire-and-forget on publish.
    """

    def __init__(self) -> None:
        self._subscribers: dict[EventType, list[EventHandler]] = defaultdict(list)

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        self._subscribers[event_type].append(handler)
        logger.debug("Subscribed %s to %s", handler.__class__.__name__, event_type)

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        self._subscribers[event_type] = [
            h for h in self._subscribers[event_type] if h is not handler
        ]

    def publish(self, event: Event) -> None:
        handlers = self._subscribers.get(event.type, [])
        logger.info("Publishing %s to %d handler(s)", event.type, len(handlers))
        for handler in handlers:
            try:
                handler(event)
            except Exception as exc:
                logger.error("Handler %s failed: %s", handler.__class__.__name__, exc)

    def on(self, *event_types: EventType) -> Callable:
        """Decorator shorthand for function-based subscribers."""
        def decorator(fn: Callable[[Event], None]) -> Callable:
            class _FnHandler(EventHandler):
                def handle(self, event: Event) -> None:
                    fn(event)

            handler = _FnHandler()
            for et in event_types:
                self.subscribe(et, handler)
            return fn
        return decorator


# --------------------------------------------------------------------------- #
# Concrete Handlers
# --------------------------------------------------------------------------- #

class LoggingHandler(EventHandler):
    def handle(self, event: Event) -> None:
        logger.info("[EVENT] %s | source=%s | %s", event.type, event.source, event.payload)


class HighRiskAlertHandler(EventHandler):
    """Fires an alert when wildfire risk score exceeds threshold."""

    def __init__(self, threshold: float = 0.75) -> None:
        self.threshold = threshold

    def handle(self, event: Event) -> None:
        if event.type != EventType.PREDICTION_MADE:
            return
        score = event.payload.get("risk_score", 0.0)
        if score >= self.threshold:
            alert_event = Event(
                type=EventType.HIGH_RISK_DETECTED,
                payload={**event.payload, "threshold": self.threshold},
                source="HighRiskAlertHandler",
            )
            logger.warning("🔥 HIGH RISK DETECTED: score=%.3f location=%s", score,
                           event.payload.get("location"))
            # In production: push to PagerDuty / Slack / GCP Pub/Sub


class ModelDriftHandler(EventHandler):
    """Tracks rolling prediction distribution to detect model drift."""

    def __init__(self, window: int = 100, drift_threshold: float = 0.15) -> None:
        self._scores: list[float] = []
        self._baseline_mean: float | None = None
        self.window = window
        self.drift_threshold = drift_threshold

    def handle(self, event: Event) -> None:
        if event.type != EventType.PREDICTION_MADE:
            return
        score = event.payload.get("risk_score")
        if score is None:
            return
        self._scores.append(score)
        if len(self._scores) == self.window:
            mean = sum(self._scores) / self.window
            if self._baseline_mean is None:
                self._baseline_mean = mean
                logger.info("Baseline prediction mean set: %.3f", mean)
            elif abs(mean - self._baseline_mean) > self.drift_threshold:
                logger.warning("⚠️ MODEL DRIFT detected: baseline=%.3f current=%.3f",
                               self._baseline_mean, mean)
            self._scores.clear()
