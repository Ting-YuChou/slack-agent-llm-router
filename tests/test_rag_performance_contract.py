from scripts.rag_performance_contract import evaluate_contract


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
