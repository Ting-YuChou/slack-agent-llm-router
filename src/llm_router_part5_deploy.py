"""
LLM Router Part 5: Deployment Configurations
Generates Docker, Kubernetes, and CI/CD configurations for production deployment
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any
import yaml

logger = logging.getLogger(__name__)


def generate_deployment_configs(output_dir: str = "."):
    """Generate all deployment configurations"""
    output_path = Path(output_dir)

    # Create deployment directories
    (output_path / "docker").mkdir(exist_ok=True)
    (output_path / "k8s").mkdir(exist_ok=True)
    (output_path / ".github" / "workflows").mkdir(parents=True, exist_ok=True)

    # Generate configurations
    generate_dockerfile(output_path / "docker" / "Dockerfile")
    generate_docker_compose(output_path / "docker-compose.yml")
    generate_docker_compose_prod(output_path / "docker-compose.prod.yml")
    generate_kubernetes_manifests(output_path / "k8s")
    generate_helm_chart(output_path / "helm")
    generate_github_actions(output_path / ".github" / "workflows")
    generate_terraform_config(output_path / "terraform")

    logger.info(f"Deployment configurations generated in {output_path}")


def generate_dockerfile(output_path: Path):
    """Generate production Dockerfile"""
    dockerfile_content = """# Multi-stage Dockerfile for LLM Router Platform
# Production-optimized with security best practices

# Build stage
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=latest

# Install system dependencies for building
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \\
    && pip install --no-cache-dir -r /tmp/requirements.txt

# Production stage
FROM python:3.11-slim as production

# Labels for metadata
LABEL maintainer="LLM Router Team" \\
      org.opencontainers.image.title="LLM Router Platform" \\
      org.opencontainers.image.description="Production-grade multi-model LLM routing platform" \\
      org.opencontainers.image.created=$BUILD_DATE \\
      org.opencontainers.image.revision=$VCS_REF \\
      org.opencontainers.image.version=$VERSION

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \\
    curl \\
    procps \\
    && rm -rf /var/lib/apt/lists/* \\
    && apt-get clean

# Create non-root user for security
RUN groupadd -r llmrouter && useradd -r -g llmrouter llmrouter

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=llmrouter:llmrouter . /app/

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/models \\
    && chown -R llmrouter:llmrouter /app

# Switch to non-root user
USER llmrouter

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8080/health || exit 1

# Expose ports
EXPOSE 8080 8000 8501

# Set environment variables
ENV PYTHONPATH=/app \\
    PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    LOG_LEVEL=INFO

# Default command
CMD ["python", "main.py", "start", "--config", "config/config.yaml"]
"""

    output_path.write_text(dockerfile_content)


def generate_docker_compose(output_path: Path):
    """Generate development docker-compose.yml"""
    compose_content = """version: '3.8'

services:
  llm-router:
    build:
      context: .
      dockerfile: docker/Dockerfile
    ports:
      - "8080:8080"    # API server
      - "8000:8000"    # Prometheus metrics
      - "8501:8501"    # Streamlit UI
    environment:
      - LOG_LEVEL=DEBUG
      - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
      - CLICKHOUSE_HOST=clickhouse
      - REDIS_HOST=redis
    volumes:
      - ./config:/app/config
      - ./logs:/app/logs
      - ./data:/app/data
    depends_on:
      - kafka
      - clickhouse
      - redis
    networks:
      - llm-router-network

  # Kafka for stream processing
  zookeeper:
    image: confluentinc/cp-zookeeper:7.4.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    volumes:
      - zookeeper_data:/var/lib/zookeeper/data
    networks:
      - llm-router-network

  kafka:
    image: confluentinc/cp-kafka:7.4.0
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:29092,PLAINTEXT_HOST://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 1
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 1
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: 'true'
    volumes:
      - kafka_data:/var/lib/kafka/data
    networks:
      - llm-router-network

  # ClickHouse for analytics
  clickhouse:
    image: clickhouse/clickhouse-server:23.8
    ports:
      - "8123:8123"
      - "9000:9000"
    environment:
      CLICKHOUSE_DB: llm_router
      CLICKHOUSE_USER: default
      CLICKHOUSE_PASSWORD: ""
    volumes:
      - clickhouse_data:/var/lib/clickhouse
      - ./clickhouse/schema.sql:/docker-entrypoint-initdb.d/init.sql
    networks:
      - llm-router-network

  # Redis for caching
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    networks:
      - llm-router-network

  # Prometheus for metrics
  prometheus:
    image: prom/prometheus:v2.45.0
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
    networks:
      - llm-router-network

  # Grafana for visualization
  grafana:
    image: grafana/grafana:10.0.0
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana:/etc/grafana/provisioning
    networks:
      - llm-router-network

  # vLLM server for local inference
  vllm-server:
    image: vllm/vllm-openai:latest
    ports:
      - "8001:8000"
    command:
      - --model
      - mistralai/Mistral-7B-Instruct-v0.2
      - --served-model-name
      - mistral-7b
    volumes:
      - ./models:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    networks:
      - llm-router-network

volumes:
  kafka_data:
  zookeeper_data:
  clickhouse_data:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  llm-router-network:
    driver: bridge
"""

    output_path.write_text(compose_content)


def generate_docker_compose_prod(output_path: Path):
    """Generate production docker-compose.yml"""
    compose_prod_content = """version: '3.8'

services:
  llm-router:
    image: llm-router:${VERSION:-latest}
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 10s
        order: start-first
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
    ports:
      - "8080:8080"
    environment:
      - LOG_LEVEL=INFO
      - KAFKA_BOOTSTRAP_SERVERS=kafka-1:9092,kafka-2:9092,kafka-3:9092
      - CLICKHOUSE_HOST=clickhouse-cluster
      - REDIS_HOST=redis-cluster
    secrets:
      - openai_api_key
      - anthropic_api_key
      - slack_bot_token
    configs:
      - source: llm_router_config
        target: /app/config/config.yaml
    volumes:
      - logs_volume:/app/logs
    networks:
      - llm-router-prod

  # Load balancer
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl/certs
    depends_on:
      - llm-router
    networks:
      - llm-router-prod

  # Kafka cluster
  kafka-1:
    image: confluentinc/cp-kafka:7.4.0
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper-1:2181,zookeeper-2:2181,zookeeper-3:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka-1:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 3
      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 2
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 3
    volumes:
      - kafka_1_data:/var/lib/kafka/data
    networks:
      - llm-router-prod

  kafka-2:
    image: confluentinc/cp-kafka:7.4.0
    environment:
      KAFKA_BROKER_ID: 2
      KAFKA_ZOOKEEPER_CONNECT: zookeeper-1:2181,zookeeper-2:2181,zookeeper-3:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka-2:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 3
      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 2
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 3
    volumes:
      - kafka_2_data:/var/lib/kafka/data
    networks:
      - llm-router-prod

  kafka-3:
    image: confluentinc/cp-kafka:7.4.0
    environment:
      KAFKA_BROKER_ID: 3
      KAFKA_ZOOKEEPER_CONNECT: zookeeper-1:2181,zookeeper-2:2181,zookeeper-3:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka-3:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 3
      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 2
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 3
    volumes:
      - kafka_3_data:/var/lib/kafka/data
    networks:
      - llm-router-prod

secrets:
  openai_api_key:
    external: true
  anthropic_api_key:
    external: true
  slack_bot_token:
    external: true

configs:
  llm_router_config:
    file: ./config/config.prod.yaml

volumes:
  kafka_1_data:
  kafka_2_data:
  kafka_3_data:
  logs_volume:

networks:
  llm-router-prod:
    driver: overlay
    attachable: true
"""

    output_path.write_text(compose_prod_content)


def generate_kubernetes_manifests(output_path: Path):
    """Generate Kubernetes manifests"""
    output_path.mkdir(exist_ok=True)

    # Namespace
    namespace_yaml = """apiVersion: v1
kind: Namespace
metadata:
  name: llm-router
  labels:
    name: llm-router
"""
    (output_path / "namespace.yaml").write_text(namespace_yaml)

    # ConfigMap
    configmap_yaml = """apiVersion: v1
kind: ConfigMap
metadata:
  name: llm-router-config
  namespace: llm-router
data:
  config.yaml: |
    api:
      host: "0.0.0.0"
      port: 8080
    kafka:
      bootstrap_servers:
        - "kafka-service:9092"
    clickhouse:
      host: "clickhouse-service"
      port: 8123
    monitoring:
      prometheus_port: 8000
"""
    (output_path / "configmap.yaml").write_text(configmap_yaml)

    # Secrets
    secrets_yaml = """apiVersion: v1
kind: Secret
metadata:
  name: llm-router-secrets
  namespace: llm-router
type: Opaque
data:
  openai-api-key: ""  # Base64 encoded
  anthropic-api-key: ""  # Base64 encoded
  slack-bot-token: ""  # Base64 encoded
"""
    (output_path / "secrets.yaml").write_text(secrets_yaml)

    # Deployment
    deployment_yaml = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-router
  namespace: llm-router
  labels:
    app: llm-router
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: llm-router
  template:
    metadata:
      labels:
        app: llm-router
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: llm-router
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: llm-router
        image: llm-router:latest
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: 8080
          name: api
        - containerPort: 8000
          name: metrics
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: llm-router-secrets
              key: openai-api-key
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: llm-router-secrets
              key: anthropic-api-key
        - name: SLACK_BOT_TOKEN
          valueFrom:
            secretKeyRef:
              name: llm-router-secrets
              key: slack-bot-token
        volumeMounts:
        - name: config
          mountPath: /app/config
        - name: logs
          mountPath: /app/logs
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 1
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
      volumes:
      - name: config
        configMap:
          name: llm-router-config
      - name: logs
        emptyDir: {}
"""
    (output_path / "deployment.yaml").write_text(deployment_yaml)

    # Service
    service_yaml = """apiVersion: v1
kind: Service
metadata:
  name: llm-router-service
  namespace: llm-router
  labels:
    app: llm-router
spec:
  selector:
    app: llm-router
  ports:
  - name: api
    port: 8080
    targetPort: 8080
    protocol: TCP
  - name: metrics
    port: 8000
    targetPort: 8000
    protocol: TCP
  type: ClusterIP
"""
    (output_path / "service.yaml").write_text(service_yaml)

    # Additional manifests omitted for brevity - similar pattern continues


def generate_helm_chart(output_path: Path):
    """Generate Helm chart"""
    output_path.mkdir(exist_ok=True)
    chart_path = output_path / "llm-router"
    chart_path.mkdir(exist_ok=True)
    (chart_path / "templates").mkdir(exist_ok=True)

    # Chart.yaml
    chart_yaml = """apiVersion: v2
name: llm-router
description: A Helm chart for LLM Router Platform
type: application
version: 0.1.0
appVersion: "1.0.0"
"""
    (chart_path / "Chart.yaml").write_text(chart_yaml)

    # values.yaml
    values_yaml = """replicaCount: 3

image:
  repository: llm-router
  pullPolicy: IfNotPresent
  tag: "latest"

service:
  type: ClusterIP
  port: 8080

resources:
  limits:
    cpu: 1000m
    memory: 2Gi
  requests:
    cpu: 500m
    memory: 1Gi
"""
    (chart_path / "values.yaml").write_text(values_yaml)


def generate_github_actions(output_path: Path):
    """Generate GitHub Actions workflows"""
    # CI workflow
    ci_yaml = """name: CI Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Python 3.11
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest-xdist pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  build:
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Build Docker image
      run: |
        docker build -t llm-router:latest -f docker/Dockerfile .
"""
    (output_path / "ci.yml").write_text(ci_yaml)


def generate_terraform_config(output_path: Path):
    """Generate Terraform configurations for infrastructure"""
    output_path.mkdir(exist_ok=True)

    # Main Terraform configuration
    main_tf = """terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "llm-router-vpc"
  }
}

resource "aws_subnet" "private" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${count.index + 1}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = {
    Name = "llm-router-private-${count.index + 1}"
  }
}

data "aws_availability_zones" "available" {
  state = "available"
}
"""
    (output_path / "main.tf").write_text(main_tf)


if __name__ == "__main__":
    generate_deployment_configs()
