# S3 + SQS RAG scaling benchmark

This benchmark finds the ingestion scaling knee for 1, 2, 4, and 8 fixed-size
ECS worker tasks. It does not set an absolute throughput SLA. Upload throughput
is reported separately and excluded from ingestion latency.

## Environment contract

Run only in an isolated staging account, with S3, SQS, Redis, the API, workers,
and embedding endpoint in one AWS region. Disable ECS autoscaling for the run.
The selected task definitions must use the same image and ingestion settings,
2 vCPU and 4 GiB per task, `consumer_count=1`, and `concurrency=1`.

Use two task definitions (or two revisions of one definition):

- `pipeline`: production parser/chunker and Redis vector index with
  `rag.embedding.provider: hash`.
- `product`: production parser/chunker and Redis vector index with the staging
  BGE endpoint and production embedding batch settings.

Do not change API, Redis, or embedding capacity between worker counts. The
resulting knee describes the whole ingestion system, so ECS CPU/memory, Redis
CPU/latency, and embedding QPS/P95/errors must be sampled. Copy
`config/rag-benchmark-cloudwatch.example.json`, replace every dimension, and
point the harness at that file.

The execution identity needs the existing API access plus `sqs:GetQueueAttributes`,
`sqs:SendMessage` for resilience injection, `ecs:UpdateService`,
`ecs:DescribeServices`, `ecs:ListTasks`, `ecs:StopTask`,
`cloudwatch:GetMetricStatistics`, and `iam:PassRole` only when the selected ECS
deployment requires it. Use workload credentials; do not place AWS keys in the
manifest or command line.

## Corpus manifest

Keep the 500 source files outside the repository. The manifest is JSON and is
also not committed:

```json
{
  "documents": [
    {
      "path": "/absolute/corpus/handbook.pdf",
      "filename": "handbook.pdf",
      "format": "pdf",
      "size_bytes": 5242880,
      "checksum_sha256": "BASE64_SHA256",
      "page_count": 42
    }
  ]
}
```

Validation reads every file and rejects count, size, or checksum drift:

```bash
python scripts/benchmark_rag_s3_sqs.py validate-manifest \
  --manifest /secure/benchmark/manifest.json --jobs 500
```

## Dry run, pilot, and matrix

First create a 10-document manifest slice and run an AWS dry run. Then create a
50-document slice and record the single-worker completion time as the pilot
value. The harness validates that the manifest contains exactly `--jobs`
documents.

```bash
python scripts/benchmark_rag_s3_sqs.py run \
  --manifest /secure/benchmark/manifest-10.json --jobs 10 \
  --api-url "$RAG_BENCHMARK_API_URL" \
  --queue-url "$RAG_AWS_QUEUE_URL" --dlq-url "$RAG_AWS_DLQ_URL" \
  --ecs-cluster "$RAG_ECS_CLUSTER" --ecs-service "$RAG_ECS_SERVICE" \
  --task-definition "$RAG_PIPELINE_TASK_DEFINITION" \
  --layer pipeline --workers 1 --mode dry_run \
  --metrics-url "$RAG_METRICS_URL" \
  --cloudwatch-config /secure/benchmark/cloudwatch.json \
  --output-dir benchmark-results/dry-run

python scripts/benchmark_rag_s3_sqs.py run \
  --manifest /secure/benchmark/manifest-50.json --jobs 50 \
  --api-url "$RAG_BENCHMARK_API_URL" \
  --queue-url "$RAG_AWS_QUEUE_URL" --dlq-url "$RAG_AWS_DLQ_URL" \
  --ecs-cluster "$RAG_ECS_CLUSTER" --ecs-service "$RAG_ECS_SERVICE" \
  --task-definition "$RAG_PIPELINE_TASK_DEFINITION" \
  --layer pipeline --workers 1 --mode pilot \
  --metrics-url "$RAG_METRICS_URL" \
  --cloudwatch-config /secure/benchmark/cloudwatch.json \
  --output-dir benchmark-results/pilot
```

After both pass, the matrix command performs a 20-document warmup for every
layer/worker group, then three 500-job runs in these orders:
`1,2,4,8`, `8,4,2,1`, and `2,8,1,4`. Its timeout is
`max(30 minutes, 3 * pilot seconds * 10)`.

```bash
python scripts/benchmark_rag_s3_sqs.py matrix \
  --manifest /secure/benchmark/manifest-500.json \
  --api-url "$RAG_BENCHMARK_API_URL" \
  --queue-url "$RAG_AWS_QUEUE_URL" --dlq-url "$RAG_AWS_DLQ_URL" \
  --ecs-cluster "$RAG_ECS_CLUSTER" --ecs-service "$RAG_ECS_SERVICE" \
  --pipeline-task-definition "$RAG_PIPELINE_TASK_DEFINITION" \
  --product-task-definition "$RAG_PRODUCT_TASK_DEFINITION" \
  --pilot-seconds 123.4 --metrics-url "$RAG_METRICS_URL" \
  --cloudwatch-config /secure/benchmark/cloudwatch.json \
  --output-dir benchmark-results/matrix
```

Set `--api-key-env RAG_BENCHMARK_API_KEY` when API-key enforcement is enabled.
The value is read from the environment and is never sent to a presigned S3 URL.
`--metrics-url` must point to a Prometheus-compatible aggregate across every API
and worker task, not a process-local metrics endpoint.
CloudWatch samples use buckets overlapping the measured interval and perform a
bounded post-run backfill (120 seconds by default) for delayed datapoints. This
wait is excluded from ingestion completion time and can be changed with
`--resource-backfill-timeout-seconds`.

## Resilience runs

Run these separately with four workers and `--mode resilience`:

- Duplicate delivery: add `--duplicate-messages 25`. The run must observe all
  injections through terminal-delivery or same-generation no-op counters while
  effective duplicate commits remain zero. The harness also compares aggregate
  atomic index-commit delta with successful documents, so an extra effective
  generation commit fails the scenario even if delivery labels are incomplete.
- Worker crash: add `--stop-worker`. The harness stops one ECS task after the
  burst and verifies terminal completion and queue drain after redelivery.
- Permanent parse failures: use a five-document invalid corpus manifest and add
  `--expected-dlq-count 5`. The scenario passes only when all five jobs become
  `dead_lettered`, the source queue drains after the 900-second visibility
  cycle, and the broker DLQ contains exactly five messages.

Clear the source queue and DLQ before every scenario. Use a unique run ID,
knowledge base, and document IDs; remove the Redis/vector benchmark namespace
after exporting results. Source objects remain durable and are removed by the
configured S3 completed lifecycle.

## Outputs and interpretation

Each run writes `summary.json`, `jobs.csv`, `queue.csv`, `resources.csv`,
`queue_timeline.svg`, and a normalized `queue_resource_timeline.svg` that
combines queue, in-flight, CPU, Redis, and embedding signals. The CSV files
retain exact units and timestamps. `analyze` or `matrix` also writes
`aggregate.json` and three SVG charts for throughput, completion time, and
pooled latency.

The aggregate reports the median and range across three runs. P50/P95/P99 are
computed from all 1,500 job latencies, never by averaging run percentiles. The
knee is N when N to 2N improves docs/min by less than 30%, or 2N has less than
60% parallel efficiency relative to one worker. If neither condition occurs at
eight workers, the report says `>8`.

A normal measured run is healthy only when all 500 jobs are terminal, failure
and duplicate deltas are zero, retry rate is at most 0.5%, and both source queue
and DLQ drain to zero. Investigate every retry using `jobs.csv` before accepting
the result.
