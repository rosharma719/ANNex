"""ANNex Python client — sync and async, read/write key auth."""

from __future__ import annotations

from typing import Any

import httpx

from .errors import AnnexError

_WRITE_SUFFIXES: frozenset[str] = frozenset([
    "/v1/vectors/upsert",
    "/v1/vectors/delete",
    "/v1/train",
    "/v1/fde/index",
    "/v1/dense/index",
    "/v1/compact",
    "/v1/collections",
])


def _is_write(path: str, method: str) -> bool:
    if path == "/v1/collections" and method.upper() == "GET":
        return False
    if path.startswith("/v1/collections/"):
        rest = path[len("/v1/collections/"):]
        op_part = rest.split("/", 1)[1] if "/" in rest else rest
        canonical = f"/v1/{op_part}"
    else:
        canonical = path
    return canonical in _WRITE_SUFFIXES


def _raise_for_status(response: httpx.Response) -> None:
    if response.is_error:
        try:
            msg = response.json().get("error", response.text)
        except Exception:
            msg = response.text
        raise AnnexError(response.status_code, msg)


class CollectionClient:
    """Scopes all requests under /v1/collections/<name>."""

    def __init__(self, parent: "AnnexClient", name: str) -> None:
        self._parent = parent
        self._prefix = f"/v1/collections/{name}"

    def _path(self, suffix: str) -> str:
        return f"{self._prefix}{suffix}"

    def upsert(self, documents: list[dict]) -> dict:
        return self._parent._post(self._path("/vectors/upsert"), {"documents": documents})

    def delete(self, id: str) -> dict:
        return self._parent._post(self._path("/vectors/delete"), {"id": id})

    def retrieve(self, request: dict) -> dict:
        return self._parent._post(self._path("/retrieve"), request)

    def query(self, vectors: list[list[float]], top_k: int,
              candidates: int | None = None, ef_search: int = 256) -> dict:
        body: dict[str, Any] = {"vectors": vectors, "top_k": top_k, "ef_search": ef_search}
        if candidates is not None:
            body["candidates"] = candidates
        return self._parent._post(self._path("/query"), body)

    def plan(self, request: dict) -> dict:
        return self._parent._post(self._path("/plan"), request)

    def stats(self) -> dict:
        return self._parent._get(self._path("/stats"))

    def build_fde_ann(self, m: int = 16, ef_construct: int = 200) -> dict:
        return self._parent._post(self._path("/fde/index"), {"m": m, "ef_construct": ef_construct})

    def build_dense_ann(self, field: str, m: int = 16, ef_construct: int = 200) -> dict:
        return self._parent._post(self._path("/dense/index"),
                                   {"field": field, "m": m, "ef_construct": ef_construct})

    def compact(self) -> dict:
        return self._parent._post(self._path("/compact"), {})


class AnnexClient:
    """
    HTTP client for the ANNex vector search service.

    Parameters
    ----------
    base_url : str
        Server base URL, e.g. ``"http://localhost:8080"``.
    read_key : str | None
        Bearer token for read-only routes.
    write_key : str | None
        Bearer token for write routes (also accepted on read routes).
    timeout : float
        Request timeout in seconds (default 30).
    _transport : httpx.BaseTransport | None
        Override the HTTP transport — used in tests.
    """

    def __init__(
        self,
        base_url: str,
        *,
        read_key: str | None = None,
        write_key: str | None = None,
        timeout: float = 30.0,
        _transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._read_key = read_key
        self._write_key = write_key
        self._timeout = timeout
        self._transport = _transport
        self._sync: httpx.Client | None = None
        self._async: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # Key selection
    # ------------------------------------------------------------------

    def _auth_header(self, path: str, method: str = "POST") -> dict[str, str]:
        write = _is_write(path, method)
        key = self._write_key if write else (self._read_key or self._write_key)
        return {"Authorization": f"Bearer {key}"} if key else {}

    # ------------------------------------------------------------------
    # Sync transport
    # ------------------------------------------------------------------

    def _client(self) -> httpx.Client:
        if self._sync is None:
            kwargs: dict[str, Any] = {"base_url": self._base_url, "timeout": self._timeout}
            if self._transport is not None:
                kwargs["transport"] = self._transport
            self._sync = httpx.Client(**kwargs)
        return self._sync

    def _get(self, path: str) -> dict:
        response = self._client().get(path, headers=self._auth_header(path, "GET"))
        _raise_for_status(response)
        return response.json()

    def _post(self, path: str, body: Any) -> dict:
        response = self._client().post(path, json=body, headers=self._auth_header(path, "POST"))
        _raise_for_status(response)
        return response.json()

    def close(self) -> None:
        if self._sync:
            self._sync.close()

    # ------------------------------------------------------------------
    # Async transport
    # ------------------------------------------------------------------

    def _async_client(self) -> httpx.AsyncClient:
        if self._async is None:
            kwargs: dict[str, Any] = {"base_url": self._base_url, "timeout": self._timeout}
            if self._transport is not None:
                kwargs["transport"] = self._transport
            self._async = httpx.AsyncClient(**kwargs)
        return self._async

    async def aclose(self) -> None:
        if self._async:
            await self._async.aclose()

    async def __aenter__(self) -> "AnnexClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    async def _aget(self, path: str) -> dict:
        response = await self._async_client().get(path, headers=self._auth_header(path, "GET"))
        _raise_for_status(response)
        return response.json()

    async def _apost(self, path: str, body: Any) -> dict:
        response = await self._async_client().post(
            path, json=body, headers=self._auth_header(path, "POST")
        )
        _raise_for_status(response)
        return response.json()

    # ------------------------------------------------------------------
    # Sync public API
    # ------------------------------------------------------------------

    def upsert(self, documents: list[dict]) -> dict:
        return self._post("/v1/vectors/upsert", {"documents": documents})

    def delete(self, id: str) -> dict:
        return self._post("/v1/vectors/delete", {"id": id})

    def retrieve(self, request: dict) -> dict:
        return self._post("/v1/retrieve", request)

    def query(self, vectors: list[list[float]], top_k: int,
              candidates: int | None = None, ef_search: int = 256) -> dict:
        body: dict[str, Any] = {"vectors": vectors, "top_k": top_k, "ef_search": ef_search}
        if candidates is not None:
            body["candidates"] = candidates
        return self._post("/v1/query", body)

    def plan(self, request: dict) -> dict:
        return self._post("/v1/plan", request)

    def stats(self) -> dict:
        return self._get("/v1/stats")

    def health(self) -> dict:
        return self._get("/healthz")

    def build_fde_ann(self, m: int = 16, ef_construct: int = 200) -> dict:
        return self._post("/v1/fde/index", {"m": m, "ef_construct": ef_construct})

    def build_dense_ann(self, field: str, m: int = 16, ef_construct: int = 200) -> dict:
        return self._post("/v1/dense/index", {"field": field, "m": m, "ef_construct": ef_construct})

    def compact(self) -> dict:
        return self._post("/v1/compact", {})

    def train(self, vectors: list[list[float]], iterations: int = 20) -> dict:
        return self._post("/v1/train", {"vectors": vectors, "iterations": iterations})

    def list_collections(self) -> list[dict]:
        return self._get("/v1/collections").get("collections", [])

    def create_collection(self, name: str, config: dict) -> dict:
        return self._post("/v1/collections", {"name": name, "config": config})

    def collection(self, name: str) -> CollectionClient:
        return CollectionClient(self, name)

    # ------------------------------------------------------------------
    # Async public API — same names; caller uses `async with` + `await`
    # ------------------------------------------------------------------

    async def async_upsert(self, documents: list[dict]) -> dict:
        return await self._apost("/v1/vectors/upsert", {"documents": documents})

    async def async_delete(self, id: str) -> dict:
        return await self._apost("/v1/vectors/delete", {"id": id})

    async def async_retrieve(self, request: dict) -> dict:
        return await self._apost("/v1/retrieve", request)

    async def async_query(self, vectors: list[list[float]], top_k: int,
                          candidates: int | None = None, ef_search: int = 256) -> dict:
        body: dict[str, Any] = {"vectors": vectors, "top_k": top_k, "ef_search": ef_search}
        if candidates is not None:
            body["candidates"] = candidates
        return await self._apost("/v1/query", body)

    async def async_plan(self, request: dict) -> dict:
        return await self._apost("/v1/plan", request)

    async def async_stats(self) -> dict:
        return await self._aget("/v1/stats")

    async def async_health(self) -> dict:
        return await self._aget("/healthz")

    async def async_build_fde_ann(self, m: int = 16, ef_construct: int = 200) -> dict:
        return await self._apost("/v1/fde/index", {"m": m, "ef_construct": ef_construct})

    async def async_build_dense_ann(self, field: str, m: int = 16,
                                    ef_construct: int = 200) -> dict:
        return await self._apost("/v1/dense/index",
                                  {"field": field, "m": m, "ef_construct": ef_construct})

    async def async_compact(self) -> dict:
        return await self._apost("/v1/compact", {})

    async def async_train(self, vectors: list[list[float]], iterations: int = 20) -> dict:
        return await self._apost("/v1/train", {"vectors": vectors, "iterations": iterations})

    async def async_list_collections(self) -> list[dict]:
        return (await self._aget("/v1/collections")).get("collections", [])

    async def async_create_collection(self, name: str, config: dict) -> dict:
        return await self._apost("/v1/collections", {"name": name, "config": config})
