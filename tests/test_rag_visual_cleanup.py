import pytest

from scripts.cleanup_rag_visual_index import cleanup_legacy_visual_index
from src.rag.vector_store import RedisStackRagVectorStore


class FakeRedis:
    def __init__(self):
        self.keys = {
            b"rag:v2:visual_chunk:{kb:doc}:chunk-1",
            b"rag:visual_chunk:legacy",
            b"rag:document:v2:{kb:doc}:visual_chunks",
            b"rag:document:kb:doc:visual_chunks",
            b"rag:v2:chunk:{kb:doc}:text-1",
        }
        self.commands = []

    async def scan_iter(self, match):
        import fnmatch

        for key in sorted(self.keys):
            if fnmatch.fnmatch(key.decode(), match):
                yield key

    async def unlink(self, *keys):
        self.keys.difference_update(keys)
        return len(keys)

    async def execute_command(self, *args):
        self.commands.append(args)
        if args[0] == "FT.INFO" and args[1] in {
            "rag:idx:v2:visual_chunks",
            "rag:idx:visual_chunks",
        }:
            return [b"index_name", b"rag:idx:v2:visual_chunks"]
        if args[0] == "FT.INFO":
            raise RuntimeError("unknown index")
        return "OK"


@pytest.mark.asyncio
async def test_visual_cleanup_dry_run_does_not_modify_redis():
    client = FakeRedis()

    result = await cleanup_legacy_visual_index(client, apply=False)

    assert result["key_count"] == 4
    assert len(client.keys) == 5
    assert not any(command[0] == "FT.DROPINDEX" for command in client.commands)


@pytest.mark.asyncio
async def test_visual_cleanup_apply_deletes_only_visual_keys_and_index():
    client = FakeRedis()

    result = await cleanup_legacy_visual_index(client, apply=True)

    assert result["deleted_keys"] == 4
    assert client.keys == {b"rag:v2:chunk:{kb:doc}:text-1"}
    assert ("FT.DROPINDEX", "rag:idx:v2:visual_chunks") in client.commands
    assert ("FT.DROPINDEX", "rag:idx:visual_chunks") in client.commands


@pytest.mark.asyncio
async def test_document_cleanup_removes_v2_and_legacy_visual_sets_and_chunks():
    class DeleteRedis:
        def __init__(self):
            self.sets = {
                "rag:document:v2:{kb:doc}:visual_chunks": {b"v2-1"},
                "rag:document:kb:doc:visual_chunks": {b"legacy-1"},
            }
            self.deleted = []

        async def smembers(self, key):
            return self.sets.get(key, set())

        async def delete(self, key):
            self.deleted.append(key)
            return 1

    store = RedisStackRagVectorStore({"redis": {"key_prefix": "rag"}})
    store.client = DeleteRedis()

    count = await store._delete_visual_chunks("doc", "kb")

    assert count == 2
    assert "rag:v2:visual_chunk:{kb:doc}:v2-1" in store.client.deleted
    assert "rag:visual_chunk:kb:legacy-1" in store.client.deleted
    assert "rag:document:v2:{kb:doc}:visual_chunks" in store.client.deleted
    assert "rag:document:kb:doc:visual_chunks" in store.client.deleted
