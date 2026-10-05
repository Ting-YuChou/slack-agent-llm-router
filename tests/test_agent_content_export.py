import base64
from contextlib import contextmanager
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.export_agent_content import export_content


def test_content_export_checks_exact_bytes_and_refuses_overwrite(tmp_path):
    content = bytes([0, 255]) + "中文".encode()
    checksum = hashlib.sha256(content).hexdigest()

    @contextmanager
    def stream(*args, **kwargs):
        yield iter(
            [
                (0, 2, checksum, base64.b64encode(content[:3]).decode()),
                (1, 2, checksum, base64.b64encode(content[3:]).decode()),
            ]
        )

    destination = tmp_path / "result.bin"
    export_content(SimpleNamespace(query_rows_stream=stream), "id", destination)
    assert destination.read_bytes() == content
    with pytest.raises(FileExistsError):
        export_content(SimpleNamespace(query_rows_stream=stream), "id", destination)
    assert destination.read_bytes() == content
    assert not list(tmp_path.glob("*.partial"))


def test_incomplete_content_does_not_publish_file(tmp_path):
    @contextmanager
    def stream(*args, **kwargs):
        yield iter([(0, 2, "bad", "YQ==")])

    destination = tmp_path / "result.bin"
    with pytest.raises(ValueError):
        export_content(SimpleNamespace(query_rows_stream=stream), "id", destination)
    assert not destination.exists()
    assert not list(tmp_path.glob("*.partial"))
