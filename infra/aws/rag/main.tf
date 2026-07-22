data "aws_caller_identity" "current" {}

locals {
  name        = "${var.project_name}-${var.environment}-rag"
  bucket_name = coalesce(var.bucket_name, "${local.name}-${data.aws_caller_identity.current.account_id}")
  tags = merge(var.tags, {
    Project     = var.project_name
    Environment = var.environment
    Component   = "rag-ingestion"
    ManagedBy   = "terraform"
  })
}

resource "aws_kms_key" "rag" {
  description             = "Encryption key for ${local.name} S3 and SQS data"
  enable_key_rotation     = true
  deletion_window_in_days = 30
  tags                    = local.tags
}

resource "aws_kms_alias" "rag" {
  name          = "alias/${local.name}"
  target_key_id = aws_kms_key.rag.key_id
}

resource "aws_s3_bucket" "rag" {
  bucket = local.bucket_name
  tags   = local.tags
}

resource "aws_s3_bucket_ownership_controls" "rag" {
  bucket = aws_s3_bucket.rag.id
  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_public_access_block" "rag" {
  bucket                  = aws_s3_bucket.rag.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "rag" {
  bucket = aws_s3_bucket.rag.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "rag" {
  bucket = aws_s3_bucket.rag.id
  rule {
    bucket_key_enabled = true
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.rag.arn
      sse_algorithm     = "aws:kms"
    }
  }
}

resource "aws_s3_bucket_cors_configuration" "rag" {
  count  = length(var.cors_allowed_origins) == 0 ? 0 : 1
  bucket = aws_s3_bucket.rag.id

  cors_rule {
    allowed_headers = ["content-length", "content-type", "x-amz-checksum-sha256", "x-amz-date", "x-amz-tagging", "authorization"]
    allowed_methods = ["PUT", "HEAD"]
    allowed_origins = var.cors_allowed_origins
    expose_headers  = ["ETag", "x-amz-version-id", "x-amz-checksum-sha256"]
    max_age_seconds = 3600
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "rag" {
  bucket = aws_s3_bucket.rag.id

  depends_on = [aws_s3_bucket_versioning.rag]

  rule {
    id     = "abort-incomplete-multipart"
    status = "Enabled"
    filter {}
    abort_incomplete_multipart_upload { days_after_initiation = 1 }
  }

  rule {
    id     = "expire-pending"
    status = "Enabled"
    filter {
      tag {
        key   = "state"
        value = "pending"
      }
    }
    expiration { days = 1 }
  }

  rule {
    id     = "expire-completed"
    status = "Enabled"
    filter {
      tag {
        key   = "state"
        value = "completed"
      }
    }
    expiration { days = 1 }
  }

  rule {
    id     = "expire-completed-with-warnings"
    status = "Enabled"
    filter {
      tag {
        key   = "state"
        value = "completed_with_warnings"
      }
    }
    expiration { days = 1 }
  }

  rule {
    id     = "expire-queued"
    status = "Enabled"
    filter {
      tag {
        key   = "state"
        value = "queued"
      }
    }
    expiration { days = 14 }
  }

  rule {
    id     = "expire-failed"
    status = "Enabled"
    filter {
      tag {
        key   = "state"
        value = "failed"
      }
    }
    expiration { days = 14 }
  }

  rule {
    id     = "expire-dead-lettered"
    status = "Enabled"
    filter {
      tag {
        key   = "state"
        value = "dead_lettered"
      }
    }
    expiration { days = 14 }
  }

  rule {
    id     = "expire-noncurrent-versions"
    status = "Enabled"
    filter {}
    noncurrent_version_expiration { noncurrent_days = 14 }
  }
}

resource "aws_s3_bucket_policy" "rag" {
  bucket = aws_s3_bucket.rag.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "DenyInsecureTransport"
      Effect    = "Deny"
      Principal = "*"
      Action    = "s3:*"
      Resource  = [aws_s3_bucket.rag.arn, "${aws_s3_bucket.rag.arn}/*"]
      Condition = { Bool = { "aws:SecureTransport" = "false" } }
    }]
  })
}

resource "aws_sqs_queue" "dlq" {
  name                      = "${local.name}-dlq"
  message_retention_seconds = 1209600
  kms_master_key_id         = aws_kms_key.rag.arn
  tags                      = local.tags
}

resource "aws_sqs_queue" "source" {
  name                       = local.name
  message_retention_seconds  = 345600
  receive_wait_time_seconds  = 20
  visibility_timeout_seconds = 900
  kms_master_key_id          = aws_kms_key.rag.arn
  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.dlq.arn
    maxReceiveCount     = var.max_receive_count
  })
  tags = local.tags
}

resource "aws_sqs_queue_redrive_allow_policy" "dlq" {
  queue_url = aws_sqs_queue.dlq.id
  redrive_allow_policy = jsonencode({
    redrivePermission = "byQueue"
    sourceQueueArns   = [aws_sqs_queue.source.arn]
  })
}

resource "aws_iam_policy" "api" {
  name = "${local.name}-api"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:GetObject",
          "s3:GetObjectVersion",
          "s3:DeleteObject",
          "s3:DeleteObjectVersion",
          "s3:PutObjectTagging",
          "s3:PutObjectVersionTagging"
        ]
        Resource = "${aws_s3_bucket.rag.arn}/rag/${var.environment}/*"
      },
      {
        Effect   = "Allow"
        Action   = ["sqs:SendMessage", "sqs:GetQueueAttributes"]
        Resource = aws_sqs_queue.source.arn
      },
      {
        Effect   = "Allow"
        Action   = ["kms:Decrypt", "kms:Encrypt", "kms:GenerateDataKey"]
        Resource = aws_kms_key.rag.arn
      }
    ]
  })
  tags = local.tags
}

resource "aws_iam_policy" "worker" {
  name = "${local.name}-worker"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:GetObjectVersion",
          "s3:PutObjectTagging",
          "s3:PutObjectVersionTagging"
        ]
        Resource = "${aws_s3_bucket.rag.arn}/rag/${var.environment}/*"
      },
      {
        Effect   = "Allow"
        Action   = ["sqs:ReceiveMessage", "sqs:DeleteMessage", "sqs:ChangeMessageVisibility", "sqs:GetQueueAttributes"]
        Resource = aws_sqs_queue.source.arn
      },
      {
        Effect   = "Allow"
        Action   = ["kms:Decrypt", "kms:Encrypt", "kms:GenerateDataKey"]
        Resource = aws_kms_key.rag.arn
      }
    ]
  })
  tags = local.tags
}

resource "aws_cloudwatch_metric_alarm" "queue_age" {
  alarm_name          = "${local.name}-oldest-message"
  namespace           = "AWS/SQS"
  metric_name         = "ApproximateAgeOfOldestMessage"
  statistic           = "Maximum"
  period              = 300
  evaluation_periods  = 2
  comparison_operator = "GreaterThanThreshold"
  threshold           = 900
  alarm_actions       = var.alarm_actions
  dimensions          = { QueueName = aws_sqs_queue.source.name }
  tags                = local.tags
}

resource "aws_cloudwatch_metric_alarm" "queue_backlog" {
  alarm_name          = "${local.name}-visible-backlog"
  namespace           = "AWS/SQS"
  metric_name         = "ApproximateNumberOfMessagesVisible"
  statistic           = "Maximum"
  period              = 300
  evaluation_periods  = 2
  comparison_operator = "GreaterThanThreshold"
  threshold           = 100
  alarm_actions       = var.alarm_actions
  dimensions          = { QueueName = aws_sqs_queue.source.name }
  tags                = local.tags
}

resource "aws_cloudwatch_metric_alarm" "dlq" {
  alarm_name          = "${local.name}-dlq-not-empty"
  namespace           = "AWS/SQS"
  metric_name         = "ApproximateNumberOfMessagesVisible"
  statistic           = "Maximum"
  period              = 60
  evaluation_periods  = 1
  comparison_operator = "GreaterThanThreshold"
  threshold           = 0
  alarm_actions       = var.alarm_actions
  dimensions          = { QueueName = aws_sqs_queue.dlq.name }
  tags                = local.tags
}
