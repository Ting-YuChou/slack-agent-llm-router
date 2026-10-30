"""Round-2 review fix (X1): only header-safe request ids go in X-Request-ID."""

import pytest
from fastapi.testclient import TestClient

from test_main import (  # noqa: F401
    DummyAdmissionController,
    _admission_platform,
    patched_platform_deps,
)


@pytest.mark.parametrize("body_id", ["日本", "é", "a\nb"])
def test_a_body_id_that_is_no_header_value_is_answered_without_the_header(
    tmp_path, patched_platform_deps, body_id
):
    platform = _admission_platform(tmp_path)
    platform.services["admission"] = DummyAdmissionController()

    with TestClient(platform._create_fastapi_app()) as client:
        response = client.post(
            "/route",
            json={"query": "hello", "user_id": "u1", "request_id": body_id},
        )

    assert response.status_code == 200
    assert "x-request-id" not in response.headers
    assert response.json()["request_id"] == body_id
