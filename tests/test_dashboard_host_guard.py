"""The dashboard answers to loopback names only.

The server's documented posture is 127.0.0.1 plus same-origin, and
``stop_run`` was accepted on that basis. The posture does not survive DNS
rebinding: engineio derives its allowed origin from the request's *own*
Host header, so a page served from an attacker-controlled name whose DNS
has been rebound to 127.0.0.1 presents a matching Host and Origin and
passes the origin check. It also skips the check entirely when Origin is
absent.

Host is the header that cannot be made to read as a loopback name while
still reaching this socket, so these tests pin the guard that compares
it. They matter more as the browser gains write endpoints: without this,
the difference between "a stranger can stop your run" and "a stranger
can write into your results tree" is one added route.

The port is deliberately not part of the comparison — see
``host_is_allowed`` — so the forwarding and default-port cases below are
contract, not accident.
"""

from __future__ import annotations

import io

import pytest

from specola.dashboard.server import (
    LOOPBACK_NAMES,
    host_is_allowed,
    host_name,
)


def _env(host):
    return {} if host is None else {"HTTP_HOST": host}


@pytest.mark.parametrize("host", [
    "127.0.0.1:8090",
    "localhost:8090",
    "[::1]:8090",
    "LOCALHOST:8090",
    "  127.0.0.1:8090  ",
])
def test_loopback_spellings_are_allowed(host):
    assert host_is_allowed(_env(host)), host


@pytest.mark.parametrize("host", [
    "evil.example:8090",
    "specola.attacker.test:8090",
    "127.0.0.1.nip.io:8090",          # resolves to loopback, is not loopback
    "localhost.attacker.test:8090",
])
def test_a_rebound_attacker_host_is_refused(host):
    """The exact shape a DNS-rebinding page presents.

    Its Host and Origin agree with each other — which is why the
    same-origin check passes — but the Host is not a loopback name.
    ``127.0.0.1.nip.io`` is the sharp case: it *resolves* to 127.0.0.1
    and is still not a loopback name.
    """
    assert not host_is_allowed(_env(host)), host


def test_a_forwarded_port_that_differs_is_still_allowed():
    """``ssh -L 8091:127.0.0.1:8090`` is a supported way to view a run.

    The browser connects to the *client-side* port, so its Host header
    names a port the server never bound. Requiring the ports to match
    would 403 the whole session with an opaque message, and would buy
    nothing: an attacker able to dictate the port could dictate the
    right one just as easily.
    """
    assert host_is_allowed(_env("localhost:8091"))
    assert host_is_allowed(_env("127.0.0.1:1"))


def test_default_ports_omit_the_port_entirely():
    """Browsers drop :80 and :443 from the Host header."""
    assert host_is_allowed(_env("127.0.0.1"))
    assert host_is_allowed(_env("localhost"))


def test_absent_or_empty_host_is_refused():
    """engineio waves through a request with no Origin; this does not."""
    assert not host_is_allowed(_env(None))
    assert not host_is_allowed(_env(""))
    assert not host_is_allowed(_env("   "))


@pytest.mark.parametrize(("header", "expected"), [
    ("localhost:8090", "localhost"),
    ("127.0.0.1", "127.0.0.1"),
    ("[::1]:8090", "::1"),
    ("[::1]", "::1"),
    ("::1", "::1"),
    ("EVIL.example:80", "evil.example"),
    ("[malformed:8090", ""),
    ("", ""),
    (None, ""),
])
def test_host_name_strips_the_port_across_spellings(header, expected):
    assert host_name(header) == expected


def test_loopback_set_is_exactly_the_three_names():
    """A fourth entry would be a new trust decision, not a typo fix."""
    assert LOOPBACK_NAMES == {"127.0.0.1", "localhost", "::1"}


def _server(tmp_path):
    pytest.importorskip("flask_socketio")
    from specola.dashboard.recording import RecordingWriter
    from specola.dashboard.server import DashboardServer

    RecordingWriter(tmp_path, {"run_id": "r", "status": "completed"})
    return DashboardServer(tmp_path, mode="view")


def test_http_routes_refuse_a_forged_host(tmp_path):
    """Through the real Flask stack, not just the predicate.

    No ``start()`` call: binding a socket here would leak a live
    listener for the rest of the pytest process, since ``shutdown()``
    only clears a flag and leaves the daemon thread inside
    ``socketio.run``. The guard is stateless, so the app enforces it
    without ever being served.
    """
    client = _server(tmp_path).app.test_client()

    refused = client.get("/", headers={"Host": "evil.example:8090"})
    assert refused.status_code == 403
    assert b"loopback" in refused.data

    for host in ("127.0.0.1:8090", "localhost:9999", "127.0.0.1"):
        allowed = client.get("/", headers={"Host": host})
        assert allowed.status_code != 403, host


def test_static_files_are_guarded_too(tmp_path):
    """The guard is outside Flask entirely, so it covers static too."""
    client = _server(tmp_path).app.test_client()
    refused = client.get(
        "/static/dashboard.js", headers={"Host": "evil.example:8090"}
    )
    assert refused.status_code == 403


def _wsgi(app, path, host):
    """Call the composed WSGI app directly, bypassing Flask routing.

    The guard is a WSGI wrapper installed outside SocketIO's middleware,
    so this is the only way to exercise the transport paths that
    middleware intercepts. ``socketio.test_client`` calls handlers
    directly and never traverses ``app.wsgi_app``, so it cannot see this
    guard at all.
    """
    captured = {}

    def start_response(status, headers, exc_info=None):
        captured["status"] = status

    environ = {
        "REQUEST_METHOD": "GET",
        "PATH_INFO": path.split("?")[0],
        "QUERY_STRING": path.split("?")[1] if "?" in path else "",
        "SERVER_NAME": "127.0.0.1",
        "SERVER_PORT": "8090",
        "wsgi.url_scheme": "http",
        "wsgi.input": io.BytesIO(),
        "wsgi.errors": io.BytesIO(),
        "HTTP_HOST": host,
    }
    body = b"".join(app.wsgi_app(environ, start_response))
    return captured["status"], body


def test_the_engineio_handshake_is_guarded(tmp_path):
    """The transport handshake allocates a session before any handler.

    SocketIO intercepts /socket.io/ ahead of Flask's routing, so a
    before_request guard never sees it, and a namespace connect handler
    runs only after Engine.IO has already answered with a sid. Measured
    against a live server before this was fixed: a forged Host got 200
    and a real sid. The guard therefore sits outside SocketIO entirely.

    If this test starts passing for the wrong reason -- because the
    wrapper was removed and something inner refuses instead -- the good
    half below fails, since nothing inner would allow it either.
    """
    app = _server(tmp_path).app
    handshake = "/socket.io/?EIO=4&transport=polling"

    status, body = _wsgi(app, handshake, "evil.example:8090")
    assert status.startswith("403"), status
    assert b"sid" not in body

    status, body = _wsgi(app, handshake, "127.0.0.1:8090")
    assert status.startswith("200"), status
    assert b"sid" in body
