"""
Producer/Consumer Pattern — Async geospatial tile ingestion pipeline.
Producers enqueue raw data chunks; consumers process them concurrently.
Designed to scale horizontally with thread/process pools or Kafka in prod.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Generator, Iterable

from src.processing.features import GeoSample

logger = logging.getLogger(__name__)

_SENTINEL = object()  # Signals consumers to stop


@dataclass
class IngestionConfig:
    num_consumers: int = 4
    queue_maxsize: int = 256
    batch_size: int = 32
    retry_limit: int = 3


@dataclass
class IngestionResult:
    total_produced: int = 0
    total_consumed: int = 0
    errors: list[str] = field(default_factory=list)


class DataProducer:
    """
    Reads raw geospatial records (CSV rows, API responses, GeoTIFF tiles)
    and enqueues GeoSample objects for downstream consumers.
    """

    def __init__(self, q: queue.Queue, source: Iterable[GeoSample]) -> None:
        self._q = q
        self._source = source
        self.produced = 0

    def run(self) -> None:
        for sample in self._source:
            self._q.put(sample)
            self.produced += 1
            if self.produced % 500 == 0:
                logger.debug("Produced %d samples", self.produced)
        self._q.put(_SENTINEL)
        logger.info("Producer done. Total produced: %d", self.produced)


class DataConsumer(threading.Thread):
    """
    Pulls GeoSample objects from the shared queue, applies a transform,
    and forwards results to a sink callable (DB write, model inference, etc.).
    """

    def __init__(
        self,
        q: queue.Queue,
        transform: Callable[[GeoSample], dict],
        sink: Callable[[list[dict]], None],
        batch_size: int = 32,
        worker_id: int = 0,
    ) -> None:
        super().__init__(daemon=True, name=f"consumer-{worker_id}")
        self._q = q
        self._transform = transform
        self._sink = sink
        self._batch_size = batch_size
        self.consumed = 0
        self.errors: list[str] = []

    def run(self) -> None:
        batch: list[dict] = []
        while True:
            item = self._q.get()
            if item is _SENTINEL:
                self._q.put(_SENTINEL)  # re-queue for other consumers
                break
            try:
                result = self._transform(item)
                batch.append(result)
                self.consumed += 1
                if len(batch) >= self._batch_size:
                    self._flush(batch)
                    batch = []
            except Exception as exc:
                self.errors.append(str(exc))
                logger.error("[%s] Transform error: %s", self.name, exc)
            finally:
                self._q.task_done()

        if batch:
            self._flush(batch)
        logger.info("[%s] Done. Consumed: %d", self.name, self.consumed)

    def _flush(self, batch: list[dict]) -> None:
        try:
            self._sink(batch)
        except Exception as exc:
            self.errors.append(f"Sink error: {exc}")
            logger.error("[%s] Sink error: %s", self.name, exc)


class IngestionPipeline:
    """
    Orchestrates the full producer/consumer pipeline.

    Usage:
        pipeline = IngestionPipeline(config, source=my_data_generator())
        result = pipeline.run(transform=feature_pipeline.transform, sink=db.write_batch)
    """

    def __init__(self, config: IngestionConfig, source: Iterable[GeoSample]) -> None:
        self._config = config
        self._source = source

    def run(
        self,
        transform: Callable[[GeoSample], dict],
        sink: Callable[[list[dict]], None],
    ) -> IngestionResult:
        cfg = self._config
        q: queue.Queue = queue.Queue(maxsize=cfg.queue_maxsize)

        producer = DataProducer(q, self._source)
        consumers = [
            DataConsumer(q, transform, sink, cfg.batch_size, worker_id=i)
            for i in range(cfg.num_consumers)
        ]

        start = time.perf_counter()

        prod_thread = threading.Thread(target=producer.run, name="producer", daemon=True)
        prod_thread.start()
        for c in consumers:
            c.start()

        prod_thread.join()
        q.join()
        for c in consumers:
            c.join()

        elapsed = time.perf_counter() - start
        result = IngestionResult(
            total_produced=producer.produced,
            total_consumed=sum(c.consumed for c in consumers),
            errors=[e for c in consumers for e in c.errors],
        )
        logger.info(
            "Pipeline complete in %.2fs | produced=%d consumed=%d errors=%d",
            elapsed, result.total_produced, result.total_consumed, len(result.errors),
        )
        return result
