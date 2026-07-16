import pytest

from scripts.cleanup_v1_provider_scheduler_keys import cleanup_legacy_keys


class FakeRedis:
    def __init__(self, values, *, ttls=None, mutate_before_eval=None):
        self.values = dict(values)
        self.ttls = dict(ttls or {})
        self.mutate_before_eval = mutate_before_eval
        self.scan_patterns = []
        self.deleted = []

    async def scan_iter(self, *, match, count):
        self.scan_patterns.append((match, count))
        for key in list(self.values):
            yield key

    async def type(self, key):
        return self.values[key]

    async def pttl(self, key):
        return self.ttls.get(key, -1)

    async def eval(self, _script, _numkeys, key):
        if self.mutate_before_eval:
            self.mutate_before_eval(self, key)
        key_type = self.values.get(key, "none")
        ttl = self.ttls.get(key, -2 if key_type == "none" else -1)
        deleted = int(key_type == "zset" and ttl == -1)
        if deleted:
            self.deleted.append(key)
            self.values.pop(key, None)
        return [deleted, key_type, ttl]


@pytest.mark.asyncio
async def test_cleanup_is_dry_run_by_default_and_reports_only_legacy_zsets():
    client = FakeRedis(
        {
            "provider_scheduler:queue:openai:gpt-5": "zset",
            "provider_scheduler:queue:anthropic:claude": "string",
            "provider_scheduler:v2:queue:openai:gpt-5:order": "zset",
            "provider_scheduler:circuit:openai:gpt-5": "zset",
        }
    )

    report = await cleanup_legacy_keys(client, apply=False)

    assert client.scan_patterns == [("provider_scheduler:queue:*", 500)]
    assert report["candidates"] == ["provider_scheduler:queue:openai:gpt-5"]
    assert report["deleted"] == []
    assert report["skipped_wrong_type"] == ["provider_scheduler:queue:anthropic:claude"]
    assert client.deleted == []


@pytest.mark.asyncio
async def test_cleanup_apply_uses_actual_v2_layout_not_model_name_heuristics():
    client = FakeRedis(
        {
            "provider_scheduler:queue:openai:gpt-5": "zset",
            "provider_scheduler:queue:openai:v2": "zset",
            "provider_scheduler:v2:queue:openai:gpt-5:order": "zset",
            "provider_scheduler:retry:openai:gpt-5": "zset",
        }
    )

    report = await cleanup_legacy_keys(client, apply=True)

    assert report["deleted"] == [
        "provider_scheduler:queue:openai:gpt-5",
        "provider_scheduler:queue:openai:v2",
    ]
    assert client.deleted == [
        "provider_scheduler:queue:openai:gpt-5",
        "provider_scheduler:queue:openai:v2",
    ]


@pytest.mark.asyncio
async def test_cleanup_skips_legacy_queue_with_ttl():
    key = "provider_scheduler:queue:openai:gpt-5"
    client = FakeRedis({key: "zset"}, ttls={key: 5000})

    report = await cleanup_legacy_keys(client, apply=True)

    assert report["candidates"] == []
    assert report["skipped_with_ttl"] == [key]
    assert client.deleted == []


@pytest.mark.asyncio
@pytest.mark.parametrize(("new_type", "new_ttl"), [("string", -1), ("zset", 5000)])
async def test_cleanup_revalidates_type_and_ttl_atomically_before_delete(
    new_type, new_ttl
):
    key = "provider_scheduler:queue:openai:gpt-5"

    def mutate(client, candidate):
        client.values[candidate] = new_type
        client.ttls[candidate] = new_ttl

    client = FakeRedis({key: "zset"}, mutate_before_eval=mutate)

    report = await cleanup_legacy_keys(client, apply=True)

    assert report["candidates"] == [key]
    assert report["deleted"] == []
    assert client.deleted == []
