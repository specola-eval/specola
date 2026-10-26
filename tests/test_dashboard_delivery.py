"""LatestOnlyDelivery behavior: depth-one queue, acks, expiry."""

from specola.dashboard.delivery import LatestOnlyDelivery


class _Bundle:
    def __init__(self, epoch, seq):
        self.epoch = epoch
        self.seq = seq

    def to_wire(self):
        return {"epoch": self.epoch, "seq": self.seq}


def _make():
    sent = []
    clock = [0.0]
    delivery = LatestOnlyDelivery(
        lambda payload, ack: sent.append((payload, ack)),
        ack_timeout_s=0.25, clock=lambda: clock[0])
    return delivery, sent, clock


def test_second_offer_waits_third_replaces_pending():
    delivery, sent, _ = _make()
    delivery.offer(_Bundle(0, 0))
    delivery.offer(_Bundle(0, 1))
    delivery.offer(_Bundle(0, 2))
    assert len(sent) == 1
    assert sent[0][0]["seq"] == 0
    assert delivery.dropped == 1
    assert delivery.pending_depth == 1


def test_ack_promotes_pending():
    delivery, sent, _ = _make()
    delivery.offer(_Bundle(0, 0))
    delivery.offer(_Bundle(0, 2))
    assert delivery.acknowledge(0, 0) is True
    assert [payload["seq"] for payload, _ in sent] == [0, 2]


def test_stale_ack_ignored():
    delivery, sent, _ = _make()
    delivery.offer(_Bundle(0, 0))
    assert delivery.acknowledge(0, 99) is False
    assert delivery.acknowledge(1, 0) is False
    assert len(sent) == 1                       # nothing new sent


def test_sender_ack_callback_acknowledges():
    delivery, sent, _ = _make()
    delivery.offer(_Bundle(0, 0))
    delivery.offer(_Bundle(0, 1))
    _, ack = sent[0]
    ack()                                       # client painted
    assert [payload["seq"] for payload, _ in sent] == [0, 1]


def test_expire_after_timeout_promotes():
    delivery, sent, clock = _make()
    delivery.offer(_Bundle(0, 0))
    delivery.offer(_Bundle(0, 1))
    assert delivery.expire() is False           # not yet timed out
    clock[0] = 0.3
    assert delivery.expire() is True
    assert delivery.timed_out == 1
    assert delivery.dropped == 1
    assert sent[-1][0]["seq"] == 1


def test_reset_clears_state():
    delivery, sent, _ = _make()
    delivery.offer(_Bundle(0, 0))
    delivery.offer(_Bundle(0, 1))
    delivery.reset()
    assert delivery.pending_depth == 0
    delivery.offer(_Bundle(1, 0))               # fresh epoch sends at once
    assert sent[-1][0] == {"epoch": 1, "seq": 0}
