"""Unit tests for the Observer event bus."""

import pytest
from src.monitoring.event_bus import Event, EventBus, EventHandler, EventType


class CapturingHandler(EventHandler):
    def __init__(self):
        self.received: list[Event] = []

    def handle(self, event: Event) -> None:
        self.received.append(event)


def make_event(event_type: EventType = EventType.PREDICTION_MADE, **payload) -> Event:
    return Event(type=event_type, payload=payload, source="test")


def test_handler_receives_published_event():
    bus = EventBus()
    handler = CapturingHandler()
    bus.subscribe(EventType.PREDICTION_MADE, handler)
    bus.publish(make_event(risk_score=0.8))
    assert len(handler.received) == 1
    assert handler.received[0].payload["risk_score"] == 0.8


def test_handler_not_called_for_other_events():
    bus = EventBus()
    handler = CapturingHandler()
    bus.subscribe(EventType.HIGH_RISK_DETECTED, handler)
    bus.publish(make_event(EventType.PREDICTION_MADE))
    assert len(handler.received) == 0


def test_multiple_handlers_all_notified():
    bus = EventBus()
    h1, h2 = CapturingHandler(), CapturingHandler()
    bus.subscribe(EventType.PREDICTION_MADE, h1)
    bus.subscribe(EventType.PREDICTION_MADE, h2)
    bus.publish(make_event())
    assert len(h1.received) == 1
    assert len(h2.received) == 1


def test_unsubscribe_stops_delivery():
    bus = EventBus()
    handler = CapturingHandler()
    bus.subscribe(EventType.PREDICTION_MADE, handler)
    bus.unsubscribe(EventType.PREDICTION_MADE, handler)
    bus.publish(make_event())
    assert len(handler.received) == 0


def test_faulty_handler_does_not_crash_bus():
    class BrokenHandler(EventHandler):
        def handle(self, event: Event) -> None:
            raise RuntimeError("💥")

    bus = EventBus()
    good = CapturingHandler()
    bus.subscribe(EventType.PREDICTION_MADE, BrokenHandler())
    bus.subscribe(EventType.PREDICTION_MADE, good)
    bus.publish(make_event())  # Should not raise
    assert len(good.received) == 1
