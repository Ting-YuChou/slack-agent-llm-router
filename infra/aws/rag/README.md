# RAG S3 and SQS module

This module creates the durable source-object and work-queue resources for RAG ingestion. Attach `api_policy_arn` to the API workload and `worker_policy_arn` to the RAG worker workload, then configure `rag.storage.backend: s3` and `rag.ingestion_queue.backend: sqs` with the module outputs.

The module deliberately does not create compute, Redis, EKS, or KEDA resources.

Set `max_receive_count` to the same value as the deployed application's
`rag.ingestion_queue.max_attempts` setting.
