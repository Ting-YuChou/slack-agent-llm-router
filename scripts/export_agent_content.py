"""Export redacted dialogue or full bash content from ClickHouse with checksum verification."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import tempfile


def _write_chunks(chunks, destination):
    destination = Path(destination)
    expected_index = 0
    count = checksum = None
    digest = hashlib.sha256()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=destination.name + ".",
            suffix=".partial",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            for index, row_count, row_checksum, encoded in chunks:
                if (
                    index != expected_index
                    or count is not None
                    and (count != row_count or checksum != row_checksum)
                ):
                    raise ValueError("Missing or inconsistent content chunks")
                data = base64.b64decode(encoded, validate=True)
                output.write(data)
                digest.update(data)
                expected_index += 1
                count = row_count
                checksum = row_checksum
            if (
                count is None
                or count != expected_index
                or digest.hexdigest() != checksum
            ):
                raise ValueError("Content checksum or count mismatch")
            output.flush()
            os.fsync(output.fileno())
        # Atomically refuse an existing destination, including a symlink.
        os.link(temporary, destination)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def export_content(client, content_id, destination):
    with client.query_rows_stream(
        "SELECT chunk_index,chunk_count,sha256,data FROM agent_content_chunks FINAL WHERE content_id={id:String} ORDER BY chunk_index",
        parameters={"id": content_id},
        settings={"max_block_size": 64},
    ) as stream:
        _write_chunks(stream, destination)


def export_bash(client, run_id, tool_call_id, destination):
    parameters = {"run": run_id, "tool": tool_call_id}
    result = client.query(
        "SELECT payload_json FROM agent_events FINAL WHERE run_id={run:String} AND kind='bash_output_manifest' AND JSONExtractString(payload_json,'tool_call_id')={tool:String} ORDER BY version DESC LIMIT 1",
        parameters=parameters,
    )
    if not result.result_rows:
        raise ValueError("No complete bash manifest exists")
    manifest = json.loads(result.result_rows[0][0])
    parameters["path"] = manifest["path"]
    with client.query_rows_stream(
        "SELECT payload_json FROM agent_events FINAL WHERE run_id={run:String} AND kind='bash_output_chunk' AND JSONExtractString(payload_json,'tool_call_id')={tool:String} AND JSONExtractString(payload_json,'path')={path:String} ORDER BY JSONExtractUInt(payload_json,'chunk_index')",
        parameters=parameters,
        settings={"max_block_size": 64},
    ) as stream:
        chunks = (
            (
                payload["chunk_index"],
                manifest["chunks"],
                manifest["sha256"],
                payload["data"],
            )
            for payload in (json.loads(row[0]) for row in stream)
        )
        _write_chunks(chunks, destination)


def main():
    import clickhouse_connect

    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--content-id")
    group.add_argument("--bash-run")
    parser.add_argument("--tool-call-id")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.bash_run and not args.tool_call_id:
        parser.error("--bash-run requires --tool-call-id")
    client = clickhouse_connect.get_client(
        host=os.getenv("CLICKHOUSE_HOST", "localhost"),
        port=int(os.getenv("CLICKHOUSE_PORT", "8123")),
        username=os.getenv("CLICKHOUSE_USER", "agent_grafana"),
        password=os.environ["CLICKHOUSE_PASSWORD"],
        database=os.getenv("CLICKHOUSE_DATABASE", "default"),
    )
    try:
        if args.content_id:
            export_content(client, args.content_id, args.output)
        else:
            export_bash(client, args.bash_run, args.tool_call_id, args.output)
    finally:
        client.close()


if __name__ == "__main__":
    main()
