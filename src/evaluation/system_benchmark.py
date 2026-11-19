"""End-to-End System Performance & Reliability Benchmark (Task 30).

Measures ingestion throughput, median/p95 processing latencies, multi-session
concurrency scalability, buffer utilization, and replay recovery rates across
IndoorAI's real-time multimodal pipeline.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import dataclasses
import json
import os
import time
from typing import Any, Dict, List, Optional
import uuid

import numpy as np

from src.core.logging import setup_logger
from backend.app.services.realtime.event_models import (
    RealtimeEvent,
    RealtimeEventType,
)
from backend.app.services.realtime.realtime_pipeline import RealtimePipeline

logger = setup_logger("SystemBenchmark")


@dataclasses.dataclass
class BenchmarkResult:
    """Structured performance benchmark metrics for an evaluation run."""
    benchmark_name: str
    total_events: int
    duration_seconds: float
    throughput_events_per_sec: float
    latency_median_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_min_ms: float
    latency_max_ms: float
    concurrent_sessions: int
    success_rate_pct: float
    buffer_dropped_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "total_events": self.total_events,
            "duration_seconds": round(self.duration_seconds, 4),
            "throughput_events_per_sec": round(self.throughput_events_per_sec, 2),
            "latency_median_ms": round(self.latency_median_ms, 3),
            "latency_p95_ms": round(self.latency_p95_ms, 3),
            "latency_p99_ms": round(self.latency_p99_ms, 3),
            "latency_min_ms": round(self.latency_min_ms, 3),
            "latency_max_ms": round(self.latency_max_ms, 3),
            "concurrent_sessions": self.concurrent_sessions,
            "success_rate_pct": round(self.success_rate_pct, 2),
            "buffer_dropped_count": self.buffer_dropped_count,
        }


class SystemPerformanceBenchmark:
    """Evaluates performance, latency, throughput, and multi-session concurrency."""

    def __init__(self, pipeline: Optional[RealtimePipeline] = None) -> None:
        self.pipeline = pipeline or RealtimePipeline()

    def run_ingestion_throughput_benchmark(
        self,
        event_count: int = 1000,
        session_id: str = "BENCH-INGEST-01",
    ) -> BenchmarkResult:
        """Benchmarks raw single-session ingestion and processing throughput."""
        stream = self.pipeline.get_or_create_stream(session_id)
        now_ms = int(time.time() * 1000)

        latencies_ms: List[float] = []
        accepted_count = 0

        start_t = time.perf_counter()
        for i in range(event_count):
            ev = RealtimeEvent(
                event_id=f"EV-BENCH-{session_id}-{i:05d}",
                session_id=session_id,
                event_type=RealtimeEventType.IMU_BATCH,
                client_timestamp_utc=now_ms + i * 20,
                device_id="BENCH-DEVICE",
                payload={
                    "samples": [
                        {"sensor_type": "ACCELEROMETER", "x": 0.0, "y": 0.1, "z": 9.81},
                        {"sensor_type": "GYROSCOPE", "x": 0.01, "y": 0.0, "z": 0.02},
                    ],
                    "step": {"step_length_m": 0.72, "heading_rad": 0.15},
                },
                sequence_number=i,
            )

            t0 = time.perf_counter()
            ack, _ = self.pipeline.ingest_event(ev)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(dt_ms)

            if ack.status.value == "ACCEPTED":
                accepted_count += 1

        total_duration = time.perf_counter() - start_t
        throughput = event_count / total_duration if total_duration > 0 else 0.0

        lat_arr = np.array(latencies_ms, dtype=float)
        result = BenchmarkResult(
            benchmark_name="single_session_ingestion_throughput",
            total_events=event_count,
            duration_seconds=total_duration,
            throughput_events_per_sec=throughput,
            latency_median_ms=float(np.median(lat_arr)),
            latency_p95_ms=float(np.percentile(lat_arr, 95)),
            latency_p99_ms=float(np.percentile(lat_arr, 99)),
            latency_min_ms=float(np.min(lat_arr)),
            latency_max_ms=float(np.max(lat_arr)),
            concurrent_sessions=1,
            success_rate_pct=(accepted_count / event_count) * 100.0 if event_count > 0 else 0.0,
            buffer_dropped_count=stream.buffer.total_dropped,
        )

        logger.info(
            "Completed ingestion benchmark: %d events in %.3fs (%.1f ev/s, median lat: %.3fms, p95: %.3fms)",
            event_count, total_duration, throughput, result.latency_median_ms, result.latency_p95_ms
        )
        return result

    def run_concurrent_multi_session_benchmark(
        self,
        num_sessions: int = 5,
        events_per_session: int = 200,
    ) -> BenchmarkResult:
        """Benchmarks concurrent throughput and isolation under multithreaded session load."""
        session_ids = [f"BENCH-CONCUR-{uuid.uuid4().hex[:6]}-{i}" for i in range(num_sessions)]
        all_latencies: List[float] = []
        total_accepted = 0
        total_events = num_sessions * events_per_session

        start_t = time.perf_counter()

        def _worker_session(sid: str) -> Tuple[List[float], int]:
            stream = self.pipeline.get_or_create_stream(sid)
            worker_lats: List[float] = []
            worker_accepted = 0
            base_t = int(time.time() * 1000)

            for i in range(events_per_session):
                # Interleave IMU and Wi-Fi events
                if i % 5 == 0:
                    ev = RealtimeEvent(
                        event_id=f"EV-CONCUR-{sid}-WIFI-{i}",
                        session_id=sid,
                        event_type=RealtimeEventType.WIFI_SCAN,
                        client_timestamp_utc=base_t + i * 50,
                        device_id="CONCUR-DEVICE",
                        payload={
                            "scan_timestamp_ms": base_t + i * 50,
                            "bssid_scans": [
                                {"bssid": f"00:11:22:33:44:{k:02x}", "rssi_dbm": -60 - k * 3, "ssid": f"WAP-{k}"}
                                for k in range(4)
                            ],
                        },
                        sequence_number=i,
                    )
                else:
                    ev = RealtimeEvent(
                        event_id=f"EV-CONCUR-{sid}-IMU-{i}",
                        session_id=sid,
                        event_type=RealtimeEventType.IMU_BATCH,
                        client_timestamp_utc=base_t + i * 50,
                        device_id="CONCUR-DEVICE",
                        payload={
                            "samples": [{"sensor_type": "ACCELEROMETER", "x": 0.0, "y": 0.0, "z": 9.8}],
                            "step": {"step_length_m": 0.70, "heading_rad": 0.0},
                        },
                        sequence_number=i,
                    )

                t0 = time.perf_counter()
                ack, _ = self.pipeline.ingest_event(ev)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                worker_lats.append(dt_ms)
                if ack.status.value == "ACCEPTED":
                    worker_accepted += 1

            return worker_lats, worker_accepted

        with ThreadPoolExecutor(max_workers=num_sessions) as executor:
            futures = [executor.submit(_worker_session, sid) for sid in session_ids]
            for f in as_completed(futures):
                lats, accepted = f.result()
                all_latencies.extend(lats)
                total_accepted += accepted

        total_duration = time.perf_counter() - start_t
        throughput = total_events / total_duration if total_duration > 0 else 0.0

        lat_arr = np.array(all_latencies, dtype=float)
        total_dropped = sum(
            self.pipeline.get_stream(sid).buffer.total_dropped
            for sid in session_ids
            if self.pipeline.get_stream(sid)
        )

        result = BenchmarkResult(
            benchmark_name="concurrent_multi_session_benchmark",
            total_events=total_events,
            duration_seconds=total_duration,
            throughput_events_per_sec=throughput,
            latency_median_ms=float(np.median(lat_arr)),
            latency_p95_ms=float(np.percentile(lat_arr, 95)),
            latency_p99_ms=float(np.percentile(lat_arr, 99)),
            latency_min_ms=float(np.min(lat_arr)),
            latency_max_ms=float(np.max(lat_arr)),
            concurrent_sessions=num_sessions,
            success_rate_pct=(total_accepted / total_events) * 100.0 if total_events > 0 else 0.0,
            buffer_dropped_count=total_dropped,
        )

        # Cleanup test benchmark sessions
        for sid in session_ids:
            self.pipeline.remove_stream(sid)

        logger.info(
            "Completed concurrent benchmark (%d sessions): %d events in %.3fs (%.1f ev/s, median lat: %.3fms, p95: %.3fms)",
            num_sessions, total_events, total_duration, throughput, result.latency_median_ms, result.latency_p95_ms
        )
        return result

    def run_replay_speed_benchmark(
        self,
        history_size: int = 500,
        session_id: str = "BENCH-REPLAY-01",
    ) -> Dict[str, Any]:
        """Measures client reconnection replay retrieval throughput."""
        stream = self.pipeline.get_or_create_stream(session_id)
        first_event_id = None

        for i in range(history_size):
            eid = f"EV-REP-{session_id}-{i:04d}"
            if i == 0:
                first_event_id = eid
            ev = RealtimeEvent(
                event_id=eid,
                session_id=session_id,
                event_type=RealtimeEventType.IMU_BATCH,
                device_id="BENCH-DEVICE",
                payload={"step": {"step_length_m": 0.7}},
            )
            stream.replay.record_processed_item(
                event=ev,
                outgoing_updates=[{"type": "LOCALIZATION_UPDATE", "x": 1.0, "y": 2.0}],
            )

        t0 = time.perf_counter()
        replayed_items, is_gap = stream.replay.get_replay_since(first_event_id)
        dt = time.perf_counter() - t0

        speed_items_per_sec = len(replayed_items) / dt if dt > 0 else 0.0

        self.pipeline.remove_stream(session_id)

        return {
            "history_size": history_size,
            "replayed_count": len(replayed_items),
            "retrieval_time_seconds": round(dt, 6),
            "replay_throughput_items_per_sec": round(speed_items_per_sec, 2),
            "is_gap_detected": is_gap,
        }

    def run_full_suite(self, output_filepath: Optional[str] = None) -> Dict[str, Any]:
        """Runs the complete performance and reliability benchmark suite."""
        logger.info("Executing IndoorAI System Performance Benchmark Suite...")
        ingest_res = self.run_ingestion_throughput_benchmark(event_count=500)
        concur_res = self.run_concurrent_multi_session_benchmark(num_sessions=5, events_per_session=100)
        replay_res = self.run_replay_speed_benchmark(history_size=500)

        suite_summary = {
            "timestamp_epoch_s": int(time.time()),
            "system": "IndoorAI Real-Time Positioning Platform",
            "suite_version": "1.0.0",
            "benchmarks": {
                "single_session_ingestion": ingest_res.to_dict(),
                "concurrent_multi_session": concur_res.to_dict(),
                "reconnection_replay": replay_res,
            },
            "sla_compliance": {
                "latency_median_target_ms": 10.0,
                "latency_median_achieved_ms": ingest_res.latency_median_ms,
                "latency_median_passed": ingest_res.latency_median_ms < 10.0,
                "latency_p95_target_ms": 50.0,
                "latency_p95_achieved_ms": ingest_res.latency_p95_ms,
                "latency_p95_passed": ingest_res.latency_p95_ms < 50.0,
                "throughput_target_events_per_sec": 100.0,
                "throughput_achieved_events_per_sec": ingest_res.throughput_events_per_sec,
                "throughput_passed": ingest_res.throughput_events_per_sec > 100.0,
                "success_rate_target_pct": 99.0,
                "success_rate_achieved_pct": ingest_res.success_rate_pct,
                "success_rate_passed": ingest_res.success_rate_pct >= 99.0,
            },
        }

        if output_filepath:
            os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
            with open(output_filepath, "w", encoding="utf-8") as f:
                json.dump(suite_summary, f, indent=2)
            logger.info("Saved benchmark suite summary to: %s", output_filepath)

        return suite_summary


if __name__ == "__main__":
    benchmark = SystemPerformanceBenchmark()
    summary = benchmark.run_full_suite()
    print(json.dumps(summary, indent=2))
