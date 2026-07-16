import pytest

from scripts.rag_performance_contract import (
    RedisOperationCounter,
    evaluate_contract,
)


def test_rag_performance_contract_rejects_each_regression():
    result = {
        "embedding_request_reduction_percent": 89.9,
        "redis_wait_reduction_percent": 89.9,
        "ingestion_throughput_improvement": 2.99,
        "retrieval_p95_improvement_percent": 29.9,
        "upload_peak_rss_delta_bytes": 64 * 1024 * 1024 + 1,
        "errors": 1,
    }

    failures = evaluate_contract(result)

    assert len(failures) == 6


def test_rag_performance_contract_accepts_thresholds():
    result = {
        "embedding_request_reduction_percent": 90.0,
        "redis_wait_reduction_percent": 90.0,
        "ingestion_throughput_improvement": 3.0,
        "retrieval_p95_improvement_percent": 30.0,
        "upload_peak_rss_delta_bytes": 64 * 1024 * 1024,
        "errors": 0,
    }

    assert evaluate_contract(result) == []


class _Pipeline:
    async def execute(self):
        return [1]


class _Redis:
    def pipeline(self, **_kwargs):
        return _Pipeline()

    async def eval(self, *_args):
        return 1

    async def hset(self, *_args, **_kwargs):
        return 1


@pytest.mark.asyncio
async def test_redis_operation_counter_measures_actual_await_boundaries():
    counter = RedisOperationCounter(_Redis())

    pipeline = counter.pipeline(transaction=False)
    await pipeline.execute()
    await counter.eval("return 1", 0)
    await counter.hset("key", mapping={"field": "value"})

    assert counter.network_waits == 3
    assert counter.operations == {"pipeline.execute": 1, "eval": 1, "hset": 1}
