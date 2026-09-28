"""HTTP/process lifecycle shared by real-server benchmarks and smoke tests."""

import json
import selectors
import subprocess
import urllib.error
import urllib.request
from contextlib import contextmanager


def http(base, route, body=None, timeout=600, *, method=None):
    data = None if body is None else json.dumps(body, allow_nan=False).encode()
    request = urllib.request.Request(
        base + route, data, {"content-type": "application/json"}, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        with error:
            detail = error.read().decode(errors="replace")
        raise RuntimeError(f"{route}: HTTP {error.code}: {detail}") from error


@contextmanager
def annex_server(
    binary, directory, dimension, *, centroids=256, durability="fsync", extra=()
):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "server.log").open("a") as log:
        process = subprocess.Popen(
            [
                str(binary),
                "--path",
                str(directory / "index"),
                "--dimension",
                str(dimension),
                "--centroids",
                str(centroids),
                "--probes",
                "8",
                "--durability",
                durability,
                "--listen",
                "127.0.0.1:0",
                *extra,
            ],
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
        )
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                if not selector.select(timeout=60):
                    raise TimeoutError(
                        f"server did not become ready; see {directory / 'server.log'}"
                    )
                line = process.stdout.readline().strip()
            if not line.startswith("multivector listening on http://"):
                raise RuntimeError(
                    f"server failed to start; see {directory / 'server.log'}"
                )
            base = line.split(" on ", 1)[1]
            http(base, "/healthz", timeout=5)
            yield base
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            process.stdout.close()
