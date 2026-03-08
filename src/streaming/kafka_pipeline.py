"""
Kafka Streaming Pipeline — production-grade replacement for the in-memory queue.

The in-memory producer/consumer works fine for a single process.
At X scale you need Kafka: persistent, distributed, replayable, multi-consumer.

Architecture:
    GeoSample events → KafkaProducer → topic: geo.samples
    KafkaConsumer group → feature extraction → topic: geo.features
    KafkaConsumer group → ML inference     → topic: geo.predictions

Local dev: docker compose up (see infra/docker-compose.yml — Kafka service added)
Prod:      Google Cloud Pub/Sub or Confluent Cloud (Kafka-compatible)

Install: pip install kafka-python
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict
from typing import Callable, Iterator

from src.processing.features import GeoSample

logger = logging.getLogger(__name__)

TOPIC_SAMPLES = "geo.samples"
TOPIC_FEATURES = "geo.features"
TOPIC_PREDICTIONS = "geo.predictions"


# --------------------------------------------------------------------------- #
# Serialization helpers
# --------------------------------------------------------------------------- #

def serialize_sample(sample: GeoSample) -> bytes:
    return json.dumps(asdict(sample)).encode("utf-8")


def deserialize_sample(data: bytes) -> GeoSample:
    return GeoSample(**json.loads(data.decode("utf-8")))


def serialize_dict(d: dict) -> bytes:
    return json.dumps(d).encode("utf-8")


def deserialize_dict(data: bytes) -> dict:
    return json.loads(data.decode("utf-8"))


# --------------------------------------------------------------------------- #
# Producer
# --------------------------------------------------------------------------- #

class GeoSampleProducer:
    """
    Publishes GeoSample events to Kafka.
    Falls back to a no-op stub if Kafka is unavailable (unit tests / CI).
    """

    def __init__(self, bootstrap_servers: str = "localhost:9092") -> None:
        self._producer = None
        self._bootstrap = bootstrap_servers
        self._connect()

    def _connect(self) -> None:
        try:
            from kafka import KafkaProducer
            self._producer = KafkaProducer(
                bootstrap_servers=self._bootstrap,
                value_serializer=lambda v: v,
                acks="all",                  # wait for all replicas
                retries=5,
                linger_ms=10,                # micro-batch for throughput
            )
            logger.info("KafkaProducer connected to %s", self._bootstrap)
        except Exception as e:
            logger.warning("Kafka unavailable (%s) — using no-op stub.", e)

    def publish(self, sample: GeoSample, key: str | None = None) -> None:
        if self._producer is None:
            logger.debug("(stub) Would publish sample at %.4f, %.4f", sample.latitude, sample.longitude)
            return
        key_bytes = key.encode() if key else None
        self._producer.send(TOPIC_SAMPLES, value=serialize_sample(sample), key=key_bytes)

    def publish_batch(self, samples: list[GeoSample]) -> None:
        for sample in samples:
            self.publish(sample)
        if self._producer:
            self._producer.flush()
            logger.info("Flushed %d samples to Kafka topic '%s'", len(samples), TOPIC_SAMPLES)

    def close(self) -> None:
        if self._producer:
            self._producer.close()


# --------------------------------------------------------------------------- #
# Consumer
# --------------------------------------------------------------------------- #

class GeoSampleConsumer(threading.Thread):
    """
    Reads GeoSample events from Kafka, applies a transform, and forwards
    results to a sink (DB write, prediction API, next Kafka topic).

    Each consumer is a thread; scale out by adding more consumer instances
    within the same consumer_group — Kafka handles partition assignment.
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        consumer_group: str = "bellwether-feature-workers",
        topic: str = TOPIC_SAMPLES,
        transform: Callable[[GeoSample], dict] | None = None,
        sink: Callable[[dict], None] | None = None,
        worker_id: int = 0,
    ) -> None:
        super().__init__(daemon=True, name=f"kafka-consumer-{worker_id}")
        self._bootstrap = bootstrap_servers
        self._group = consumer_group
        self._topic = topic
        self._transform = transform or (lambda s: asdict(s))
        self._sink = sink or (lambda d: logger.info("sink: %s", d))
        self._running = False
        self._consumer = None
        self.processed = 0
        self.errors: list[str] = []

    def _connect(self) -> bool:
        try:
            from kafka import KafkaConsumer
            self._consumer = KafkaConsumer(
                self._topic,
                bootstrap_servers=self._bootstrap,
                group_id=self._group,
                auto_offset_reset="earliest",
                enable_auto_commit=True,
                value_deserializer=lambda v: v,
                session_timeout_ms=30_000,
                heartbeat_interval_ms=10_000,
            )
            logger.info("[%s] Connected to Kafka topic '%s'", self.name, self._topic)
            return True
        except Exception as e:
            logger.error("[%s] Kafka connection failed: %s", self.name, e)
            return False

    def run(self) -> None:
        if not self._connect():
            logger.warning("[%s] Running in stub mode — no messages will be consumed.", self.name)
            return

        self._running = True
        logger.info("[%s] Listening...", self.name)

        try:
            for message in self._consumer:
                if not self._running:
                    break
                try:
                    sample = deserialize_sample(message.value)
                    result = self._transform(sample)
                    self._sink(result)
                    self.processed += 1
                    if self.processed % 100 == 0:
                        logger.debug("[%s] Processed %d messages", self.name, self.processed)
                except Exception as e:
                    self.errors.append(str(e))
                    logger.error("[%s] Processing error: %s", self.name, e)
        finally:
            self._consumer.close()
            logger.info("[%s] Stopped. Processed: %d, Errors: %d", self.name, self.processed, len(self.errors))

    def stop(self) -> None:
        self._running = False


# --------------------------------------------------------------------------- #
# Pipeline orchestrator
# --------------------------------------------------------------------------- #

class KafkaPipeline:
    """
    High-level orchestrator: spins up N consumer threads and a producer.

    Usage:
        pipeline = KafkaPipeline(bootstrap_servers="localhost:9092", n_consumers=4)
        pipeline.start(transform=feature_pipeline.transform, sink=db.write)
        pipeline.producer.publish_batch(samples)
        pipeline.stop()
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        n_consumers: int = 4,
        consumer_group: str = "bellwether-workers",
    ) -> None:
        self._bootstrap = bootstrap_servers
        self._n = n_consumers
        self._group = consumer_group
        self.producer = GeoSampleProducer(bootstrap_servers)
        self._consumers: list[GeoSampleConsumer] = []

    def start(
        self,
        transform: Callable[[GeoSample], dict],
        sink: Callable[[dict], None],
    ) -> None:
        self._consumers = [
            GeoSampleConsumer(self._bootstrap, self._group,
                              transform=transform, sink=sink, worker_id=i)
            for i in range(self._n)
        ]
        for c in self._consumers:
            c.start()
        logger.info("KafkaPipeline started with %d consumers.", self._n)

    def stop(self) -> None:
        for c in self._consumers:
            c.stop()
        self.producer.close()
        logger.info("KafkaPipeline stopped.")
