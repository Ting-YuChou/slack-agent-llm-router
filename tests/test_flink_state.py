"""A8: Flink keeps durable state, stable operator ids and a resuming submitter."""

import importlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from flink import analytics_job

ROOT = Path(__file__).resolve().parents[1]
FLINK_SERVICES = ("flink-jobmanager", "flink-taskmanager")
CHECKPOINT_MOUNT = "flink_checkpoints:/opt/flink/checkpoints"
UID_REQUIRED = ("source", "process", "sink")


class _Stream:
    def __init__(self, nodes, kind="source", fn=None, parent=None):
        self.kind = kind
        self.fn = fn
        self.parent = parent
        self.uid_value = None
        self.sink_name = None
        nodes.append(self)
        self._nodes = nodes

    def _then(self, kind, fn):
        return _Stream(self._nodes, kind, fn, self)

    def map(self, fn, output_type=None):
        return self._then("map", fn)

    def filter(self, fn):
        return self._then("filter", fn)

    def key_by(self, fn):
        return self._then("key_by", fn)

    def process(self, fn, output_type=None):
        return self._then("process", fn)

    def assign_timestamps_and_watermarks(self, strategy):
        return self._then("watermarks", strategy)

    def add_sink(self, sink):
        return self._then("sink", sink)

    def name(self, name):
        self.sink_name = name
        return self

    def uid(self, uid):
        assert self.kind != "key_by", "uid belongs on operators, not key_by"
        self.uid_value = uid
        return self


class _Env:
    def __init__(self):
        self.nodes = []
        self.parallelism = None
        self.checkpoint_interval_ms = None

    def set_parallelism(self, parallelism):
        self.parallelism = parallelism

    def enable_checkpointing(self, interval_ms):
        self.checkpoint_interval_ms = interval_ms

    def add_source(self, consumer, name):
        return _Stream(self.nodes)


def _assert_unique_uids(nodes, expected_count):
    required = [node for node in nodes if node.kind in UID_REQUIRED]
    assert len(required) == expected_count
    missing = [
        (node.kind, node.sink_name, node.fn) for node in required if not node.uid_value
    ]
    assert missing == []
    uids = [node.uid_value for node in nodes if node.uid_value]
    assert len(uids) == len(set(uids))
    return uids


def test_analytics_job_names_every_source_stateful_operator_and_sink(monkeypatch):
    env = _Env()
    monkeypatch.setattr(
        analytics_job,
        "StreamExecutionEnvironment",
        SimpleNamespace(get_execution_environment=lambda: env),
    )
    monkeypatch.setattr(analytics_job, "FlinkKafkaConsumer", lambda **kwargs: kwargs)
    monkeypatch.setattr(analytics_job, "FlinkKafkaProducer", lambda **kwargs: kwargs)
    monkeypatch.setattr(analytics_job, "SimpleStringSchema", lambda: None)
    monkeypatch.setattr(analytics_job, "WatermarkStrategy", None)

    analytics_job.create_flink_job({"flink": {"checkpointing_interval_ms": 5000}})

    # 1 source, 6 keyed process operators, 7 sinks.
    uids = _assert_unique_uids(env.nodes, 14)
    mapper = next(
        node
        for node in env.nodes
        if isinstance(node.fn, analytics_job.InferenceCompletedMapper)
    )
    assert mapper.uid_value
    assert "user-policy-state" in uids and "session-policy-state" in uids
    assert env.checkpoint_interval_ms == 5000


def _import_enrichment_job(monkeypatch):
    class _Function:
        pass

    stubs = {
        "pyflink": {},
        "pyflink.common": {},
        "pyflink.common.serialization": {"SimpleStringSchema": lambda: None},
        "pyflink.common.typeinfo": {
            "Types": SimpleNamespace(
                PICKLED_BYTE_ARRAY=lambda: None, STRING=lambda: None
            )
        },
        "pyflink.datastream": {"StreamExecutionEnvironment": object},
        "pyflink.datastream.connectors": {},
        "pyflink.datastream.connectors.kafka": {
            "FlinkKafkaConsumer": lambda **kwargs: kwargs,
            "FlinkKafkaProducer": lambda **kwargs: kwargs,
        },
        "pyflink.datastream.functions": {
            "FilterFunction": _Function,
            "MapFunction": _Function,
        },
    }
    for name, attributes in stubs.items():
        module = types.ModuleType(name)
        for attribute, value in attributes.items():
            setattr(module, attribute, value)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "flink.job", raising=False)
    return importlib.import_module("flink.job")


def test_enrichment_job_reads_flink_config_and_names_its_operators(monkeypatch):
    job = _import_enrichment_job(monkeypatch)
    env = _Env()
    monkeypatch.setattr(
        job,
        "StreamExecutionEnvironment",
        SimpleNamespace(get_execution_environment=lambda: env),
    )

    job.create_flink_job(
        {"flink": {"parallelism": 2, "checkpointing_interval_ms": 5000}}
    )

    assert env.checkpoint_interval_ms == 5000
    assert env.parallelism == 2
    # 1 source and 3 sinks; the stateless enrichment mapper is named as well.
    _assert_unique_uids(env.nodes, 4)
    mapper = next(
        node for node in env.nodes if isinstance(node.fn, job.QueryEnrichmentMapper)
    )
    assert mapper.uid_value


def test_enrichment_job_top_level_keys_still_win(monkeypatch):
    job = _import_enrichment_job(monkeypatch)
    env = _Env()
    monkeypatch.setattr(
        job,
        "StreamExecutionEnvironment",
        SimpleNamespace(get_execution_environment=lambda: env),
    )

    job.create_flink_job(
        {
            "parallelism": 1,
            "checkpoint_interval_ms": 1000,
            "flink": {"parallelism": 2, "checkpointing_interval_ms": 5000},
        }
    )

    assert (env.parallelism, env.checkpoint_interval_ms) == (1, 1000)


def _compose():
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))


def _flink_properties(service):
    text = service["environment"]["FLINK_PROPERTIES"]
    return dict(
        (key.strip(), value.strip())
        for key, value in (
            line.split(":", 1) for line in text.splitlines() if line.strip()
        )
    )


@pytest.mark.parametrize("service_name", FLINK_SERVICES)
def test_flink_services_keep_rocksdb_checkpoints_on_a_named_volume(service_name):
    compose = _compose()
    service = compose["services"][service_name]
    properties = _flink_properties(service)

    assert properties["state.backend.type"] == "rocksdb"
    assert properties["state.checkpoints.dir"] == "file:///opt/flink/checkpoints"
    assert properties["state.savepoints.dir"].startswith(
        "file:///opt/flink/checkpoints/"
    )
    assert (
        properties["execution.checkpointing.externalized-checkpoint-retention"]
        == "RETAIN_ON_CANCELLATION"
    )
    assert service["restart"] == "unless-stopped"
    assert CHECKPOINT_MOUNT in service["volumes"]
    assert "flink_checkpoints" in compose["volumes"]


def test_flink_submitter_service_resumes_jobs_after_topics_exist():
    services = _compose()["services"]
    submitter = services["flink-jobs"]
    command = " ".join(submitter["command"])

    assert "/workspace/flink/submitter.py" in command
    assert "--watch" in command
    assert "/workspace/config/config.compose.yaml" in command
    assert submitter["depends_on"]["flink-jobmanager"]["condition"] == "service_healthy"
    assert submitter["depends_on"]["chat-topics"]["condition"] == (
        "service_completed_successfully"
    )
    assert CHECKPOINT_MOUNT in submitter["volumes"]
    assert "./config:/workspace/config:ro" in submitter["volumes"]
    assert submitter.get("ports") is None
    assert "healthcheck" in services["flink-jobmanager"]


# --- submitter ---------------------------------------------------------------

from flink import submitter  # noqa: E402


def _checkpoint(root, job_id, number, complete=True):
    path = root / job_id / f"chk-{number}"
    path.mkdir(parents=True)
    if complete:
        (path / "_metadata").write_bytes(b"x")
    return path


def test_latest_checkpoint_picks_the_newest_complete_one(tmp_path):
    _checkpoint(tmp_path, "old", 3)
    newest = _checkpoint(tmp_path, "new", 12)
    _checkpoint(tmp_path, "new", 13, complete=False)
    _checkpoint(tmp_path, "new", 9)

    assert submitter.latest_checkpoint(tmp_path, ["new", "old"]) == newest
    assert submitter.latest_checkpoint(tmp_path, ["missing"]) is None


def test_job_config_carries_the_yaml_flink_section():
    app_config = yaml.safe_load(
        (ROOT / "config" / "config.compose.yaml").read_text(encoding="utf-8")
    )
    analytics = next(job for job in submitter.JOBS if job.slug == "inference-analytics")

    config = submitter.build_job_config(app_config, analytics, parallelism=1)

    assert config["flink"] == app_config["flink"]
    assert config["flink"]["analytics"]["provider_guardrails"]
    assert config["kafka"]["topics"] == app_config["kafka"]["topics"]
    assert config["kafka"]["bootstrap_servers"] == ["kafka:29092"]
    assert config["kafka"]["consumer_group"] == "flink-inference-analytics"
    assert config["parallelism"] == 1


class _Cluster:
    def __init__(self, running=(), fail_restore=False):
        self.jobs = [
            {"jid": f"jid-{index}", "name": name, "state": "RUNNING"}
            for index, name in enumerate(running)
        ]
        self.fail_restore = fail_restore
        self.commands = []

    def fetch_json(self, url):
        assert url.endswith("/jobs/overview")
        return {"jobs": self.jobs}

    def run(self, command, env):
        self.commands.append((command, json.loads(env["FLINK_CONFIG_JSON"])))
        if self.fail_restore and "-s" in command:
            return SimpleNamespace(returncode=1, stdout="", stderr="bad checkpoint")
        job_id = f"new-{len(self.commands)}"
        return SimpleNamespace(
            returncode=0,
            stdout=f"Job has been submitted with JobID {job_id}\n",
            stderr="",
        )


def _reconcile(tmp_path, cluster, app_config=None):
    return submitter.reconcile(
        app_config or {"flink": {"checkpointing_interval_ms": 5000}, "kafka": {}},
        jobmanager="flink-jobmanager:8081",
        checkpoint_root=tmp_path,
        fetch_json=cluster.fetch_json,
        run=cluster.run,
    )


def test_reconcile_resumes_missing_jobs_from_their_latest_checkpoint(tmp_path):
    analytics = next(job for job in submitter.JOBS if job.slug == "inference-analytics")
    enrichment = next(job for job in submitter.JOBS if job.slug == "request-enrichment")
    submitter.record_job_id(tmp_path, analytics.slug, "prev-analytics")
    checkpoint = _checkpoint(tmp_path, "prev-analytics", 7)
    cluster = _Cluster(running=[enrichment.name])

    result = _reconcile(tmp_path, cluster)

    assert result == {enrichment.slug: "running", analytics.slug: "restored"}
    [(command, job_config)] = cluster.commands
    assert command[:5] == ["flink", "run", "-m", "flink-jobmanager:8081", "-d"]
    assert command[command.index("-s") + 1] == str(checkpoint)
    assert command[-1].endswith("flink/analytics_job.py")
    assert job_config["flink"]["checkpointing_interval_ms"] == 5000
    assert submitter.recorded_job_ids(tmp_path, analytics.slug)[0] == "new-1"


def test_reconcile_submits_fresh_without_a_checkpoint(tmp_path):
    cluster = _Cluster()

    result = _reconcile(tmp_path, cluster)

    assert set(result.values()) == {"submitted"}
    assert all("-s" not in command for command, _ in cluster.commands)


def test_reconcile_falls_back_to_fresh_when_restore_fails(tmp_path):
    analytics = next(job for job in submitter.JOBS if job.slug == "inference-analytics")
    enrichment = next(job for job in submitter.JOBS if job.slug == "request-enrichment")
    submitter.record_job_id(tmp_path, analytics.slug, "prev")
    _checkpoint(tmp_path, "prev", 2)
    cluster = _Cluster(running=[enrichment.name], fail_restore=True)

    result = _reconcile(tmp_path, cluster)

    assert result[analytics.slug] == "submitted_fresh_after_restore_failure"
    assert ["-s" in command for command, _ in cluster.commands] == [True, False]


def test_recorded_job_ids_keep_a_short_history(tmp_path):
    for index in range(8):
        submitter.record_job_id(tmp_path, "inference-analytics", f"id-{index}")

    ids = submitter.recorded_job_ids(tmp_path, "inference-analytics")
    assert ids[0] == "id-7"
    assert len(ids) == submitter.JOB_ID_HISTORY
