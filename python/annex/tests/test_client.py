import pytest
import httpx
from annex import AnnexClient, AnnexError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class CapturingTransport(httpx.BaseTransport, httpx.AsyncBaseTransport):
    """Records the last request; works for both sync and async clients."""

    def __init__(self, status: int = 200, body: dict | None = None):
        self.last_request: httpx.Request | None = None
        self._status = status
        self._body = body or {}

    def _make_response(self) -> httpx.Response:
        import json
        return httpx.Response(
            self._status,
            content=json.dumps(self._body).encode(),
            headers={"content-type": "application/json"},
        )

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.last_request = request
        return self._make_response()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.last_request = request
        return self._make_response()


def client_with_transport(transport: CapturingTransport, **kwargs) -> AnnexClient:
    return AnnexClient("http://localhost:8080", _transport=transport, **kwargs)


# ---------------------------------------------------------------------------
# AnnexError
# ---------------------------------------------------------------------------

def test_annex_error_stores_status_and_message():
    err = AnnexError(404, "not found")
    assert err.status_code == 404
    assert err.message == "not found"
    assert "404" in str(err)
    assert "not found" in str(err)


# ---------------------------------------------------------------------------
# Auth header tests
# ---------------------------------------------------------------------------

def test_no_keys_sends_no_auth_header():
    t = CapturingTransport()
    c = client_with_transport(t)
    c.stats()
    assert "authorization" not in t.last_request.headers


def test_write_op_sends_write_key():
    t = CapturingTransport()
    c = client_with_transport(t, write_key="wk-test")
    c.upsert([])
    assert t.last_request.headers["authorization"] == "Bearer wk-test"


def test_read_op_sends_read_key_when_set():
    t = CapturingTransport(body={"matches": [], "trace": {}})
    c = client_with_transport(t, read_key="rk-test", write_key="wk-test")
    c.retrieve({"prefetch": [], "limit": 5})
    assert t.last_request.headers["authorization"] == "Bearer rk-test"


def test_read_op_falls_back_to_write_key():
    t = CapturingTransport(body={"matches": [], "trace": {}})
    c = client_with_transport(t, write_key="wk-test")
    c.retrieve({"prefetch": [], "limit": 5})
    assert t.last_request.headers["authorization"] == "Bearer wk-test"


# ---------------------------------------------------------------------------
# Error raising
# ---------------------------------------------------------------------------

def test_4xx_raises_annex_error():
    t = CapturingTransport(status=403, body={"error": "invalid key"})
    c = client_with_transport(t)
    with pytest.raises(AnnexError) as exc_info:
        c.stats()
    assert exc_info.value.status_code == 403
    assert "invalid key" in exc_info.value.message


def test_5xx_raises_annex_error():
    t = CapturingTransport(status=500, body={"error": "internal error"})
    c = client_with_transport(t)
    with pytest.raises(AnnexError):
        c.stats()


# ---------------------------------------------------------------------------
# Request routing
# ---------------------------------------------------------------------------

def test_upsert_posts_to_correct_path():
    t = CapturingTransport()
    c = client_with_transport(t)
    c.upsert([{"id": "a", "text": "hello", "metadata": {}}])
    assert t.last_request.url.path == "/v1/vectors/upsert"
    assert t.last_request.method == "POST"


def test_retrieve_posts_to_correct_path():
    t = CapturingTransport(body={"matches": [], "trace": {}})
    c = client_with_transport(t)
    c.retrieve({"prefetch": [], "limit": 5})
    assert t.last_request.url.path == "/v1/retrieve"


def test_stats_gets_correct_path():
    t = CapturingTransport()
    c = client_with_transport(t)
    c.stats()
    assert t.last_request.url.path == "/v1/stats"
    assert t.last_request.method == "GET"


# ---------------------------------------------------------------------------
# CollectionClient scoping
# ---------------------------------------------------------------------------

def test_collection_prepends_path():
    t = CapturingTransport()
    c = client_with_transport(t)
    col = c.collection("myindex")
    col.stats()
    assert t.last_request.url.path == "/v1/collections/myindex/stats"


def test_collection_upsert_sends_write_key():
    t = CapturingTransport()
    c = client_with_transport(t, write_key="wk-test")
    col = c.collection("myindex")
    col.upsert([])
    assert t.last_request.headers["authorization"] == "Bearer wk-test"


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_stats():
    t = CapturingTransport()
    async with AnnexClient("http://localhost:8080", _transport=t) as c:
        await c.async_stats()
    assert t.last_request.url.path == "/v1/stats"


@pytest.mark.asyncio
async def test_async_no_auth_header_with_no_keys():
    t = CapturingTransport()
    async with AnnexClient("http://localhost:8080", _transport=t) as c:
        await c.async_stats()
    assert "authorization" not in t.last_request.headers
