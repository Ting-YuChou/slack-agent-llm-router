variable "project_name" {
  description = "Short project identifier used in resource names."
  type        = string
  default     = "slack-llm-router"
}

variable "environment" {
  description = "Deployment environment, for example development or production."
  type        = string
}

variable "bucket_name" {
  description = "Optional globally unique bucket name. A deterministic account-scoped name is used when null."
  type        = string
  default     = null
}

variable "cors_allowed_origins" {
  description = "Origins allowed to use presigned browser PUT uploads."
  type        = list(string)
  default     = []
}

variable "alarm_actions" {
  description = "SNS topic ARNs invoked by CloudWatch alarms."
  type        = list(string)
  default     = []
}

variable "max_receive_count" {
  description = "SQS deliveries before DLQ redrive; must match rag.ingestion_queue.max_attempts."
  type        = number
  default     = 3

  validation {
    condition     = var.max_receive_count >= 1 && var.max_receive_count <= 1000 && floor(var.max_receive_count) == var.max_receive_count
    error_message = "max_receive_count must be a whole number between 1 and 1000."
  }
}

variable "tags" {
  description = "Additional tags applied to managed resources."
  type        = map(string)
  default     = {}
}
