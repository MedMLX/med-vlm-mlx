"""The HTTP API keeps request images and network exposure at trusted boundaries."""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from mlx_vlm.server.app import app
from mlx_vlm.server.cli import parse_arguments
from mlx_vlm.server.schemas import ChatRequest


@pytest.mark.parametrize(
    "source",
    [
        "https://example.com/image.png",
        "http://127.0.0.1:8081/private.png",
        "http://169.254.169.254/latest/meta-data/",
        "/tmp/private.png",
    ],
)
def test_http_image_rejects_server_side_sources(source: str) -> None:
    payload = {
        "model": "medgemma",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": source}}],
            }
        ],
    }
    with pytest.raises(ValidationError, match="base64 data:image URI"):
        ChatRequest.model_validate(payload)


def test_http_image_accepts_inline_data() -> None:
    payload = {
        "model": "medgemma",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,iVBORw0KGgo=",
                    }
                ],
            }
        ],
    }
    assert ChatRequest.model_validate(payload).messages[0].role == "user"


def test_browser_origin_cannot_access_loopback_api() -> None:
    with TestClient(app) as client:
        response = client.options(
            "/v1/chat/completions",
            headers={
                "Host": "127.0.0.1:8080",
                "Origin": "https://attacker.example",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
    assert "access-control-allow-origin" not in response.headers


def test_unauthenticated_server_rejects_host_rebinding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MLX_VLM_SERVER_API_KEY", raising=False)
    with TestClient(app) as client:
        hostile = client.get("/health", headers={"Host": "attacker.example"})
        loopback = client.get("/health", headers={"Host": "127.0.0.1"})
        monkeypatch.setenv("MLX_VLM_SERVER_API_KEY", "secret")
        authenticated = client.get(
            "/health",
            headers={"Host": "api.example", "Authorization": "Bearer secret"},
        )
    assert hostile.status_code == 400
    assert loopback.status_code == 200
    assert authenticated.status_code == 200


def test_non_loopback_bind_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLX_VLM_SERVER_API_KEY", raising=False)
    assert parse_arguments([]).host == "127.0.0.1"
    with pytest.raises(SystemExit) as blocked:
        parse_arguments(["--host", "0.0.0.0"])
    assert blocked.value.code == 2
    assert (
        parse_arguments(["--host", "0.0.0.0", "--api-key", "secret"]).api_key
        == "secret"
    )
    monkeypatch.setenv("MLX_VLM_SERVER_API_KEY", "secret")
    assert parse_arguments(["--host", "::"]).host == "::"
