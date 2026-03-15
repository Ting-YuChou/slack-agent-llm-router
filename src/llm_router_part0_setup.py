"""
LLM Router Part 0: Project Setup and Environment Configuration
Handles initial setup, directory structure, and environment validation
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Optional
import yaml
import json

from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)


class ProjectSetup:
    """Handles project initialization and environment setup"""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).absolute()
        self.required_dirs = [
            "config",
            "data/queries",
            "data/processed/routed",
            "docker",
            ".github/workflows",
            "flink",
            "kafka",
            "clickhouse/data",
            "monitoring/grafana",
            "slack/credentials",
            "streamlit_ui",
            "logs",
            "src/models",
            "src/utils",
            "tests",
        ]
        self.required_files = {
            ".gitignore": self._get_gitignore_content(),
            "README.md": self._get_readme_content(),
            "requirements.txt": self._get_requirements_content(),
            "docker/requirements.txt": self._get_docker_requirements_content(),
            "kafka/topics.json": self._get_kafka_topics_config(),
            "clickhouse/schema.sql": self._get_clickhouse_schema(),
            "monitoring/prometheus.yml": self._get_prometheus_config(),
            "monitoring/grafana/dashboard.json": self._get_grafana_dashboard(),
            "streamlit_ui/config.toml": self._get_streamlit_config(),
            ".github/workflows/ci.yml": self._get_github_actions_config(),
        }

    def setup_project_environment(self):
        """Complete project setup"""
        logger.info("Starting project environment setup...")

        # Create directory structure
        self._create_directories()

        # Create configuration files
        self._create_files()

        # Setup Python environment
        self._setup_python_environment()

        # Validate environment
        self._validate_environment()

        logger.info("Project environment setup completed successfully!")

    def _create_directories(self):
        """Create required directory structure"""
        logger.info("Creating project directories...")

        for dir_path in self.required_dirs:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {full_path}")

        # Create __init__.py files for Python packages
        python_packages = ["src", "src/models", "src/utils", "tests"]
        for package in python_packages:
            init_file = self.project_root / package / "__init__.py"
            init_file.touch(exist_ok=True)

    def _create_files(self):
        """Create required configuration files"""
        logger.info("Creating configuration files...")

        for file_path, content in self.required_files.items():
            full_path = self.project_root / file_path

            if not full_path.exists():
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content)
                logger.debug(f"Created file: {full_path}")
            else:
                logger.debug(f"File already exists: {full_path}")

    def _setup_python_environment(self):
        """Setup Python virtual environment and install dependencies"""
        logger.info("Setting up Python environment...")

        # Check if virtual environment exists
        venv_path = self.project_root / "venv"
        if not venv_path.exists():
            logger.info("Creating virtual environment...")
            subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)

        # Install requirements
        requirements_path = self.project_root / "requirements.txt"
        if requirements_path.exists():
            logger.info("Installing Python dependencies...")
            pip_path = (
                venv_path / "bin" / "pip"
                if os.name != "nt"
                else venv_path / "Scripts" / "pip.exe"
            )
            subprocess.run(
                [str(pip_path), "install", "-r", str(requirements_path)], check=True
            )

    def _validate_environment(self):
        """Validate that the environment is properly set up"""
        logger.info("Validating environment setup...")

        # Check required directories
        for dir_path in self.required_dirs:
            full_path = self.project_root / dir_path
            if not full_path.exists():
                raise FileNotFoundError(f"Required directory missing: {full_path}")

        # Check critical files
        critical_files = ["config/config.yaml", "requirements.txt"]
        for file_path in critical_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                raise FileNotFoundError(f"Critical file missing: {full_path}")

        # Validate configuration
        self._validate_config()

        logger.info("Environment validation completed successfully!")

    def _validate_config(self):
        """Validate configuration file"""
        config_path = self.project_root / "config" / "config.yaml"

        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            # Validate required sections
            required_sections = ["api", "router", "inference", "kafka", "monitoring"]
            for section in required_sections:
                if section not in config:
                    raise ValueError(
                        f"Required configuration section missing: {section}"
                    )

        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise

    def _get_gitignore_content(self) -> str:
        return """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
#  Usually these files are written by a python script from a template
#  before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
#   According to pypa/pipenv#598, it is recommended to include Pipfile.lock in version control.
#   However, in case of collaboration, if having platform-specific dependencies or dependencies
#   having no cross-platform support, pipenv may install dependencies that don't work, or not
#   install all needed dependencies.
#Pipfile.lock

# PEP 582; used by e.g. github.com/David-OConnor/pyflow
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# LLM Router specific
logs/
data/processed/
slack/credentials/
*.log
.DS_Store
.vscode/
.idea/

# Model files
models/
*.bin
*.safetensors
*.gguf

# API keys and secrets
.env.local
.env.production
**/token.txt
**/api_keys.json

# ClickHouse data
clickhouse/data/

# Kafka logs
kafka/*.log

# Docker volumes
docker/volumes/
"""

    def _get_readme_content(self) -> str:
        return """# LLM Router & Execution Platform

Production-grade multi-model deployment system for enterprise AI applications.

## 🚀 Features

- **Multi-Model Routing**: Intelligent routing between OpenAI, Anthropic, and self-hosted models
- **Context Compression**: Advanced prompt compression for long-context scenarios  
- **Real-time Monitoring**: Comprehensive metrics and observability with Prometheus/Grafana
- **Stream Processing**: High-throughput query processing with Kafka and Flink
- **Slack Integration**: Enterprise-ready Slack bot with conversation continuity
- **Production Ready**: Docker deployment, CI/CD, and enterprise-grade architecture

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Slack Bot     │    │   FastAPI Server │    │  Streamlit UI   │
│   Integration   │    │   (API Gateway)  │    │   Dashboard     │
└─────────┬───────┘    └──────────┬───────┘    └─────────────────┘
          │                       │
          └───────────────────────┼───────────────────────────────┐
                                  │                               │
                 ┌────────────────▼─────────────────┐              │
                 │         Model Router             │              │
                 │    (Query Classification &       │              │
                 │     Model Selection Logic)       │              │
                 └────────────────┬─────────────────┘              │
                                  │                               │
        ┌─────────────────────────┼─────────────────────────────┐  │
        │                         ▼                             │  │
        │            ┌─────────────────────────────┐            │  │
        │            │     Inference Engine        │            │  │
        │            │   - vLLM (Self-hosted)     │            │  │
        │            │   - OpenAI API             │            │  │
        │            │   - Anthropic API          │            │  │
        │            │   - Context Compression    │            │  │
        │            │   - Response Caching       │            │  │
        │            └─────────────────────────────┘            │  │
        │                                                       │  │
        └───────────────────────┬───────────────────────────────┘  │
                                │                                  │
                 ┌──────────────▼───────────────┐                  │
                 │        Kafka Pipeline        │                  │
                 │   - Query Ingestion          │                  │
                 │   - Response Processing      │                  │
                 │   - Metrics Collection       │                  │
                 └──────────────┬───────────────┘                  │
                                │                                  │
        ┌───────────────────────┼──────────────────────────────────┼──┐
        │                       ▼                                  │  │
        │  ┌─────────────────────────┐    ┌──────────────────────┐ │  │
        │  │     ClickHouse DB       │    │   Monitoring Stack   │ │  │
        │  │   - Query Logs          │    │   - Prometheus       │ │  │
        │  │   - Performance Metrics │    │   - Grafana          │ │  │
        │  │   - User Analytics      │    │   - Alert Manager    │ │  │
        │  └─────────────────────────┘    └──────────────────────┘ │  │
        └────────────────────────────────────────────────────────────┘  │
                                                                        │
        ┌───────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────┐
│  Flink Jobs     │
│  (Optional)     │
│  - Stream       │
│    Processing   │
└─────────────────┘
```

## 🛠️ Quick Start

### Prerequisites

- Python 3.9+
- Docker & Docker Compose
- CUDA-compatible GPU (for local model inference)
- 16GB+ RAM recommended

### Installation

1. **Clone and setup the project:**
```bash
git clone <repository-url>
cd slack-agent-llm-router
python main.py setup
```

2. **Configure environment variables:**
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

3. **Start with Docker Compose:**
```bash
docker-compose up -d
```

4. **Or run locally:**
```bash
python main.py start --dev
```

### Configuration

Edit `config/config.yaml` to customize:

- **Model Selection**: Add/remove models, set priorities
- **Routing Rules**: Define query-to-model mapping logic  
- **Resource Limits**: Set memory, GPU, and token limits
- **Integration Settings**: Configure Slack, monitoring, etc.

## 📊 Monitoring & Observability

- **Grafana Dashboard**: http://localhost:3000
- **Prometheus Metrics**: http://localhost:8000/metrics
- **Streamlit UI**: http://localhost:8501
- **API Health Check**: http://localhost:8080/health

## 🔧 Development

### Running Tests
```bash
pytest tests/ -v
```

### Code Quality
```bash
black src/ tests/
flake8 src/ tests/
mypy src/
```

### Adding New Models
1. Update `config/config.yaml` with model configuration
2. Implement model wrapper in `src/models/`
3. Add routing rules for the new model

## 🚀 Deployment

### Docker Production Deployment
```bash
python main.py deploy
docker-compose -f docker-compose.prod.yml up -d
```

### Kubernetes Deployment
```bash
kubectl apply -f k8s/
```

### CI/CD Pipeline
GitHub Actions workflow automatically:
- Runs tests and quality checks
- Builds Docker images
- Deploys to staging/production

## 📈 Performance Optimization

- **GPU Utilization**: Optimized vLLM configuration for maximum throughput
- **Context Compression**: Reduces token usage by up to 70%
- **Response Caching**: Sub-millisecond cache hits for repeated queries
- **Batch Processing**: Processes multiple queries simultaneously
- **Connection Pooling**: Efficient resource management

## 🔒 Security

- API key management with environment variables
- Rate limiting and request validation
- CORS configuration for web integration
- Audit logging for compliance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

- GitHub Issues: For bug reports and feature requests
- Documentation: [docs/](docs/) directory
- Examples: [examples/](examples/) directory

---

Built with ❤️ for enterprise AI applications
"""

    def _get_requirements_content(self) -> str:
        return """# Core framework dependencies
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0

# Async and HTTP
httpx==0.25.2
aiohttp==3.9.1
asyncio-mqtt==0.13.0

# Database and storage
clickhouse-connect==0.6.23
redis==5.0.1
sqlalchemy==2.0.23

# Message queue and streaming
aiokafka==0.8.11
kafka-python==2.0.2
apache-flink==1.18.0

# LLM and AI libraries
openai==1.3.7
anthropic==0.7.8
transformers==4.36.2
torch==2.1.1
vllm==0.2.2
sentence-transformers==2.2.2

# Monitoring and observability
prometheus-client==0.19.0
grafana-api==1.0.3
sentry-sdk==1.38.0

# Slack integration
slack-sdk==3.26.1
slack-bolt==1.18.1

# UI and visualization
streamlit==1.28.2
plotly==5.17.0
pandas==2.1.4

# Configuration and utilities
pyyaml==6.0.1
click==8.1.7
python-dotenv==1.0.0
tenacity==8.2.3

# Development and testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
black==23.11.0
flake8==6.1.0
mypy==1.7.1

# Security
cryptography==41.0.8
python-jose[cryptography]==3.3.0
bcrypt==4.1.2

# Additional utilities
rich==13.7.0
structlog==23.2.0
jinja2==3.1.2
"""

    def _get_docker_requirements_content(self) -> str:
        return """# Docker-optimized requirements for production
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
httpx==0.25.2
clickhouse-connect==0.6.23
redis==5.0.1
aiokafka==0.8.11
openai==1.3.7
anthropic==0.7.8
vllm==0.2.2
prometheus-client==0.19.0
slack-sdk==3.26.1
pyyaml==6.0.1
tenacity==8.2.3
structlog==23.2.0
"""

    def _get_kafka_topics_config(self) -> str:
        return json.dumps(
            {
                "topics": [
                    {
                        "name": "llm-queries",
                        "partitions": 6,
                        "replication_factor": 1,
                        "config": {
                            "retention.ms": "604800000",
                            "compression.type": "gzip",
                        },
                    },
                    {
                        "name": "llm-responses",
                        "partitions": 6,
                        "replication_factor": 1,
                        "config": {
                            "retention.ms": "604800000",
                            "compression.type": "gzip",
                        },
                    },
                    {
                        "name": "llm-metrics",
                        "partitions": 3,
                        "replication_factor": 1,
                        "config": {
                            "retention.ms": "2592000000",
                            "compression.type": "gzip",
                        },
                    },
                    {
                        "name": "llm-errors",
                        "partitions": 3,
                        "replication_factor": 1,
                        "config": {
                            "retention.ms": "2592000000",
                            "compression.type": "gzip",
                        },
                    },
                ]
            },
            indent=2,
        )

    def _get_clickhouse_schema(self) -> str:
        return """-- LLM Router ClickHouse Schema
-- Production-grade table definitions

-- Create database
CREATE DATABASE IF NOT EXISTS llm_router;

-- Query logs table
CREATE TABLE IF NOT EXISTS llm_router.query_logs (
    query_id String,
    timestamp DateTime64(3),
    user_id String,
    user_tier Enum8('free' = 1, 'premium' = 2, 'enterprise' = 3),
    query_text String,
    query_type Enum8('general' = 1, 'code_generation' = 2, 'analysis' = 3, 'summarization' = 4),
    selected_model String,
    token_count_input UInt32,
    token_count_output UInt32,
    latency_ms UInt32,
    cost_usd Float64,
    status Enum8('success' = 1, 'error' = 2, 'timeout' = 3),
    error_message String,
    context_compressed Boolean,
    compression_ratio Float32,
    cached_response Boolean
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, user_id, query_id)
TTL timestamp + INTERVAL 90 DAY;

-- System metrics table
CREATE TABLE IF NOT EXISTS llm_router.system_metrics (
    timestamp DateTime64(3),
    service String,
    metric_name String,
    metric_value Float64,
    labels Map(String, String)
) ENGINE = ReplacingMergeTree()
PARTITION BY toYYYYMMDD(timestamp)
ORDER BY (service, metric_name, timestamp)
TTL timestamp + INTERVAL 30 DAY;

-- Model performance table
CREATE TABLE IF NOT EXISTS llm_router.model_performance (
    timestamp DateTime64(3),
    model_name String,
    requests_count UInt32,
    success_rate Float32,
    avg_latency_ms Float32,
    avg_tokens_per_second Float32,
    error_count UInt32,
    total_cost_usd Float64,
    gpu_utilization Float32,
    memory_usage_gb Float32
) ENGINE = ReplacingMergeTree()
PARTITION BY toYYYYMMDD(timestamp)
ORDER BY (model_name, timestamp)
TTL timestamp + INTERVAL 60 DAY;

-- User analytics table
CREATE TABLE IF NOT EXISTS llm_router.user_analytics (
    user_id String,
    date Date,
    total_queries UInt32,
    total_tokens UInt32,
    total_cost_usd Float64,
    avg_query_length Float32,
    most_used_model String,
    query_types Array(String),
    satisfaction_score Float32
) ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (user_id, date)
TTL date + INTERVAL 180 DAY;

-- Create materialized views for real-time analytics
CREATE MATERIALIZED VIEW IF NOT EXISTS llm_router.hourly_metrics
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMMDD(hour)
ORDER BY (hour, model_name)
AS SELECT
    toStartOfHour(timestamp) as hour,
    selected_model as model_name,
    count() as requests,
    sum(token_count_input + token_count_output) as total_tokens,
    avg(latency_ms) as avg_latency,
    sum(cost_usd) as total_cost
FROM llm_router.query_logs
GROUP BY hour, model_name;

-- Indexes for better query performance
ALTER TABLE llm_router.query_logs ADD INDEX idx_user_id user_id TYPE bloom_filter GRANULARITY 1;
ALTER TABLE llm_router.query_logs ADD INDEX idx_model selected_model TYPE bloom_filter GRANULARITY 1;
ALTER TABLE llm_router.query_logs ADD INDEX idx_status status TYPE bloom_filter GRANULARITY 1;
"""

    def _get_prometheus_config(self) -> str:
        return """# Prometheus configuration for LLM Router monitoring
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'llm-router-api'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'llm-router-inference'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'vllm-server'
    static_configs:
      - targets: ['localhost:8001']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'kafka-exporter'
    static_configs:
      - targets: ['localhost:9308']
    scrape_interval: 15s

  - job_name: 'clickhouse-exporter'
    static_configs:
      - targets: ['localhost:9363']
    scrape_interval: 30s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
    scrape_interval: 15s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
"""

    def _get_grafana_dashboard(self) -> str:
        return json.dumps(
            {
                "dashboard": {
                    "id": None,
                    "title": "LLM Router Platform Dashboard",
                    "tags": ["llm", "ai", "production"],
                    "timezone": "browser",
                    "panels": [
                        {
                            "id": 1,
                            "title": "Request Rate",
                            "type": "stat",
                            "targets": [
                                {
                                    "expr": "rate(llm_requests_total[5m])",
                                    "legendFormat": "Requests/sec",
                                }
                            ],
                            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                        },
                        {
                            "id": 2,
                            "title": "Response Time",
                            "type": "stat",
                            "targets": [
                                {
                                    "expr": "histogram_quantile(0.95, rate(llm_request_duration_seconds_bucket[5m]))",
                                    "legendFormat": "95th percentile",
                                }
                            ],
                            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                        },
                        {
                            "id": 3,
                            "title": "Model Usage Distribution",
                            "type": "piechart",
                            "targets": [
                                {
                                    "expr": "sum by (model) (rate(llm_requests_total[1h]))",
                                    "legendFormat": "{{model}}",
                                }
                            ],
                            "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8},
                        },
                    ],
                    "time": {"from": "now-6h", "to": "now"},
                    "refresh": "30s",
                }
            },
            indent=2,
        )

    def _get_streamlit_config(self) -> str:
        return """[theme]
base = "dark"
primaryColor = "#FF6B6B"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"

[server]
port = 8501
address = "0.0.0.0"
headless = true
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[logger]
level = "info"
"""

    def _get_github_actions_config(self) -> str:
        return """name: LLM Router CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      redis:
        image: redis:7-alpine
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379

    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python 3.9
      uses: actions/setup-python@v4
      with:
        python-version: 3.9
    
    - name: Cache pip dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest-xdist
    
    - name: Run code quality checks
      run: |
        black --check src/ tests/
        flake8 src/ tests/
        mypy src/
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=src --cov-report=xml -n auto
    
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

  security-scan:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Run Bandit security scan
      run: |
        pip install bandit
        bandit -r src/ -f json -o bandit-report.json
    
    - name: Run Safety check
      run: |
        pip install safety
        safety check --json --output safety-report.json || true

  docker-build:
    runs-on: ubuntu-latest
    needs: [test, security-scan]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Login to DockerHub
      if: github.event_name != 'pull_request'
      uses: docker/login-action@v3
      with:
        username: ${{ secrets.DOCKERHUB_USERNAME }}
        password: ${{ secrets.DOCKERHUB_TOKEN }}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        file: ./docker/Dockerfile
        push: ${{ github.event_name != 'pull_request' }}
        tags: |
          llm-router:latest
          llm-router:${{ github.sha }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy:
    runs-on: ubuntu-latest
    needs: docker-build
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Deploy to staging
      run: |
        echo "Deploying to staging environment"
        # Add your deployment commands here
    
    - name: Run integration tests
      run: |
        echo "Running integration tests"
        # Add integration test commands here
    
    - name: Deploy to production
      if: success()
      run: |
        echo "Deploying to production environment"
        # Add production deployment commands here
"""


def setup_project_environment():
    """Main function to setup project environment"""
    setup_logging()
    setup = ProjectSetup()
    setup.setup_project_environment()


if __name__ == "__main__":
    setup_project_environment()
