"""Per-client depth-one media delivery with paint acknowledgements.

Ported from HRC/monitor/dashboard/delivery.py (2026-07-30). Bundles are
duck-typed: anything exposing ``epoch``, ``seq`` and ``to_wire()``
(specola's ``protocol.Bundle``) works.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Callable, Optional, Tuple

Sender = Callable[[dict, Callable[..., None]], None]


class LatestOnlyDelivery:
    """Keep one in-flight bundle and replace one pending bundle."""

    def __init__(
        self,
        sender: Sender,
        *,
        ack_timeout_s: float = 0.25,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._sender = sender
        self._ack_timeout_s = float(ack_timeout_s)
        self._clock = clock
        self._lock = threading.Lock()
        self._inflight: Optional[Any] = None
        self._inflight_offered_at: Optional[float] = None
        self._inflight_sent_at: Optional[float] = None
        self._pending: Optional[Tuple[Any, float]] = None
        self._ack_ms = deque(maxlen=2048)
        self.emitted = 0
        self.dropped = 0
        self.timed_out = 0

    @property
    def pending_depth(self) -> int:
        with self._lock:
            return int(self._pending is not None)

    @property
    def ack_p95_ms(self) -> Optional[float]:
        with self._lock:
            values = sorted(self._ack_ms)
        if not values:
            return None
        index = max(0, min(len(values) - 1, round(0.95 * (len(values) - 1))))
        return round(values[index], 3)

    def offer(self, bundle: Any) -> None:
        send = None
        offered_at = self._clock()
        with self._lock:
            if self._inflight is None:
                self._mark_inflight(bundle, offered_at)
                send = bundle
            else:
                if self._pending is not None:
                    self.dropped += 1
                self._pending = (bundle, offered_at)
        if send is not None:
            self._send(send)

    def acknowledge(self, epoch: int, seq: int) -> bool:
        send = None
        with self._lock:
            current = self._inflight
            if (
                current is None
                or current.epoch != int(epoch)
                or current.seq != int(seq)
            ):
                return False
            assert self._inflight_offered_at is not None
            self._ack_ms.append(
                (self._clock() - self._inflight_offered_at) * 1000.0
            )
            send = self._retire_and_promote()
        if send is not None:
            self._send(send)
        return True

    def expire(self) -> bool:
        send = None
        with self._lock:
            if (
                self._inflight is None
                or self._inflight_sent_at is None
                or self._clock() - self._inflight_sent_at
                < self._ack_timeout_s
            ):
                return False
            self.timed_out += 1
            self.dropped += 1
            send = self._retire_and_promote()
        if send is not None:
            self._send(send)
        return True

    def _retire_and_promote(self) -> Optional[Any]:
        """Clear the in-flight slot and promote the pending bundle, if
        any. Caller holds the lock; the returned bundle must be sent
        after releasing it."""
        self._inflight = None
        self._inflight_offered_at = None
        self._inflight_sent_at = None
        if self._pending is None:
            return None
        send, offered_at = self._pending
        self._pending = None
        self._mark_inflight(send, offered_at)
        return send

    def reset(self) -> None:
        with self._lock:
            self._inflight = None
            self._inflight_offered_at = None
            self._inflight_sent_at = None
            self._pending = None

    def _mark_inflight(self, bundle: Any, offered_at: float) -> None:
        self._inflight = bundle
        self._inflight_offered_at = offered_at
        self._inflight_sent_at = self._clock()
        self.emitted += 1

    def _send(self, bundle: Any) -> None:
        def ack(*_args) -> None:
            self.acknowledge(bundle.epoch, bundle.seq)

        self._sender(bundle.to_wire(), ack)
