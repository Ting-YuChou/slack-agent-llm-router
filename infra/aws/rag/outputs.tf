output "bucket_name" { value = aws_s3_bucket.rag.id }
output "bucket_arn" { value = aws_s3_bucket.rag.arn }
output "queue_url" { value = aws_sqs_queue.source.id }
output "queue_arn" { value = aws_sqs_queue.source.arn }
output "dlq_url" { value = aws_sqs_queue.dlq.id }
output "kms_key_arn" { value = aws_kms_key.rag.arn }
output "api_policy_arn" { value = aws_iam_policy.api.arn }
output "worker_policy_arn" { value = aws_iam_policy.worker.arn }
