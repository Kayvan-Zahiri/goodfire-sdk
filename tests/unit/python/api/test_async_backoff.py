import asyncio
import json

import httpx
import pytest

from goodfire.api.exceptions import RateLimitException
from goodfire.api.features.client import AsyncFeaturesAPI
from goodfire.api.utils import AsyncHTTPWrapper


@pytest.fixture
def cooperative_http(monkeypatch):
    requests = []
    delays = []
    heartbeat = asyncio.Event()
    original_sleep = asyncio.sleep
    original_client = httpx.AsyncClient

    async def tick():
        heartbeat.set()

    async def cooperative_sleep(delay):
        delays.append(delay)
        task = asyncio.create_task(tick())
        await original_sleep(0)
        await task

    def blocking_sleep(_delay):
        raise AssertionError("blocking sleep on the async event loop")

    def install(statuses, payload):
        def respond(request):
            index = len(requests)
            assert index < len(statuses), "retry limit exceeded"
            requests.append(request)
            return httpx.Response(statuses[index], json=payload)

        monkeypatch.setattr(
            "goodfire.api.utils.httpx.AsyncClient",
            lambda: original_client(transport=httpx.MockTransport(respond)),
        )

    monkeypatch.setattr("goodfire.api.utils.time.sleep", blocking_sleep)
    monkeypatch.setattr(asyncio, "sleep", cooperative_sleep)
    return install, requests, delays, heartbeat


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["get", "post", "put", "delete"])
@pytest.mark.parametrize("statuses", [[200], [429, 429, 200], [429, 429, 429]])
async def test_async_backoff_yields_and_preserves_request(
    cooperative_http, method, statuses
):
    install, requests, delays, heartbeat = cooperative_http
    install(statuses, {"ok": True})
    wrapper = AsyncHTTPWrapper(inital_backoff_time=2, max_retries=2)
    kwargs = {"headers": {"X-Test": "preserved"}, "timeout": 7}
    if method == "get":
        kwargs["params"] = {"query": "example"}
    elif method in ("post", "put"):
        kwargs["json"] = {"payload": "example"}
    if statuses[-1] == 429:
        with pytest.raises(RateLimitException):
            await getattr(wrapper, method)("https://example.com/test", **kwargs)
    else:
        result = await getattr(wrapper, method)("https://example.com/test", **kwargs)
        assert result.json() == {"ok": True}
    assert len(requests) == len(statuses)
    assert delays == (
        [] if len(statuses) == 1 else [2, 4] if method == "post" else [1, 2]
    )
    assert heartbeat.is_set() == bool(delays)
    for request in requests:
        assert request.headers["X-Test"] == "preserved"
        assert request.extensions["timeout"]["read"] == 7
        if method == "get":
            assert request.url.params["query"] == "example"
        elif method in ("post", "put"):
            assert json.loads(request.content) == {"payload": "example"}


@pytest.mark.asyncio
@pytest.mark.parametrize("statuses", [[200], [429, 200]])
async def test_public_feature_search_remains_responsive(cooperative_http, statuses):
    install, requests, delays, heartbeat = cooperative_http
    install(statuses, {"features": []})
    client = AsyncFeaturesAPI("test-key", base_url="https://example.com")
    result = await client.search("query", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    assert len(result) == 0
    assert len(requests) == len(statuses)
    assert delays == ([] if len(statuses) == 1 else [1])
    assert heartbeat.is_set() == bool(delays)
