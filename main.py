#!/usr/bin/env python3
"""
LLM Router & Execution Platform - Main Entry Point
Production-grade multi-model deployment system for Slack AI

This is the unified entry point that orchestrates all system components:
- Model Router Service
- Inference Engine
- Kafka Pipeline
- Monitoring Stack
- Slack Bot Integration
"""

import asyncio
import logging
import multiprocessing as mp
import os
import signal
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import uvicorn
import yaml
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from prometheus_client import start_http_server

# Import all our components
from src.llm_router_part1_router import ModelRouter
from src.llm_router_part2_inference import InferenceEngine
from src.llm_router_part3_pipeline import KafkaIngestionPipeline
from src.llm_router_part4_monitor import MonitoringService
from src.utils.logger import setup_logging
from src.utils.metrics import SYSTEM_METRICS
from src.utils.schema import QueryRequest
from slack.bot import SlackBot


DEFAULT_CONFIG_PATH = "config/config.yaml"
CONFIG_ENV_VAR = "LLM_ROUTER_CONFIG"


def resolve_config_path(config_path: Optional[str] = None) -> str:
    """Resolve config path from explicit input or environment."""
    return config_path or os.getenv(CONFIG_ENV_VAR, DEFAULT_CONFIG_PATH)


def get_api_worker_count(config: Dict[str, Any], dev: bool = False) -> int:
    """Return the configured API worker count."""
    if dev:
        return 1

    workers = config.get("performance", {}).get("workers", {}).get("api_workers", 1)
    try:
        return max(1, int(workers))
    except (TypeError, ValueError):
        return 1


class LLMRouterPlatform:
    """Main orchestrator for the LLM Router Platform"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = resolve_config_path(config_path)
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config()
        self.services: Dict[str, Any] = {}
        self.processes: List[mp.Process] = []
        self.shutdown_event = asyncio.Event()

        # Setup logging
        setup_logging(
            log_level=self.config.get("logging", {}).get("level", "INFO"),
            log_file=self.config.get("logging", {}).get("file", "logs/llm_router.log")
        )
        self.logger = logging.getLogger(__name__)

    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logging.getLogger(__name__).error(
                f"Configuration file not found: {self.config_path}"
            )
            sys.exit(1)
        except yaml.YAMLError as e:
            logging.getLogger(__name__).error(f"Error parsing configuration: {e}")
            sys.exit(1)

    def _service_enabled(self, section: str) -> bool:
        """Return whether a top-level config section is enabled."""
        return bool(self.config.get(section, {}).get("enabled", False))

    def _build_pipeline_config(self) -> Dict[str, Any]:
        """Build pipeline config from top-level Kafka and ClickHouse sections."""
        pipeline_config = dict(self.config.get("kafka", {}))
        pipeline_config["clickhouse"] = dict(self.config.get("clickhouse", {}))
        pipeline_config["pipeline"] = dict(self.config.get("pipeline", {}))
        return pipeline_config

    async def _initialize_core_services(self):
        """Initialize the router and inference services required by the API."""
        self.services["router"] = ModelRouter(
            config=self.config.get("router", {})
        )
        await self.services["router"].initialize()

        self.services["inference"] = InferenceEngine(
            config=self.config.get("inference", {}),
            router=self.services["router"]
        )
        await self.services["inference"].initialize()

    async def _initialize_background_services(self):
        """Initialize enabled background services."""
        if self._service_enabled("pipeline"):
            self.services["pipeline"] = KafkaIngestionPipeline(
                config=self._build_pipeline_config()
            )
            await self.services["pipeline"].initialize()

        if self._service_enabled("monitoring"):
            self.services["monitoring"] = MonitoringService(
                config=self.config.get("monitoring", {})
            )
            await self.services["monitoring"].initialize()

        if self._service_enabled("slack"):
            self.services["slack_bot"] = SlackBot(
                config=self.config.get("slack", {}),
                inference_engine=self.services["inference"]
            )
            await self.services["slack_bot"].initialize()

    async def _initialize_services(
        self, include_api: bool = True, include_background: bool = True
    ):
        """Initialize platform services for the selected runtime mode."""
        self.logger.info("Initializing LLM Router Platform services...")

        needs_core_services = include_api or (
            include_background and self._service_enabled("slack")
        )
        if needs_core_services:
            await self._initialize_core_services()

        if include_background:
            await self._initialize_background_services()
        self.logger.info("Service initialization complete")

    def _start_metrics_server(self):
        """Start the Prometheus metrics server when monitoring is enabled."""
        if not self._service_enabled("monitoring"):
            return

        metrics_port = self.config.get("monitoring", {}).get("prometheus_port", 8000)
        start_http_server(metrics_port)
        self.logger.info(f"Prometheus metrics server started on port {metrics_port}")

    async def _start_services(
        self, include_api: bool = True, include_background: bool = True
    ):
        """Start selected platform services."""
        self.logger.info("Starting platform services...")

        tasks = []

        if include_background:
            self._start_metrics_server()

            if "pipeline" in self.services:
                tasks.append(
                    asyncio.create_task(
                        self.services["pipeline"].start(),
                        name="kafka_pipeline"
                    )
                )

            if "monitoring" in self.services:
                tasks.append(
                    asyncio.create_task(
                        self.services["monitoring"].start(),
                        name="monitoring_service"
                    )
                )

            if "slack_bot" in self.services:
                tasks.append(
                    asyncio.create_task(
                        self.services["slack_bot"].start(),
                        name="slack_bot"
                    )
                )

        if include_api:
            api_config = self.config.get("api", {})
            tasks.append(
                asyncio.create_task(
                    self._start_api_server(api_config),
                    name="api_server"
                )
            )

        if not tasks:
            self.logger.warning("No services selected to start; exiting")
            return

        self.logger.info("Selected services started")

        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            self.logger.error(f"Service error: {e}")
            await self._shutdown_services()

    async def _start_api_server(self, api_config: Dict):
        """Start the FastAPI server"""
        app = self._create_fastapi_app()

        config = uvicorn.Config(
            app,
            host=api_config.get("host", "0.0.0.0"),
            port=api_config.get("port", 8080),
            loop="asyncio",
            log_level=api_config.get("log_level", "info")
        )

        server = uvicorn.Server(config)
        await server.serve()

    def _record_api_metrics(
        self, endpoint: str, method: str, status_code: int,
        duration_seconds: float, error_type: Optional[str] = None
    ):
        """Record request-level API metrics."""
        SYSTEM_METRICS.requests_total.labels(
            endpoint=endpoint,
            method=method,
            status=str(status_code),
        ).inc()
        SYSTEM_METRICS.request_duration.labels(
            endpoint=endpoint,
            method=method,
        ).observe(duration_seconds)

        if error_type:
            SYSTEM_METRICS.errors_total.labels(
                component="api",
                error_type=error_type,
            ).inc()

    def _create_fastapi_app(self, lifespan=None) -> FastAPI:
        """Create FastAPI application with all endpoints"""
        app = FastAPI(
            title="LLM Router & Execution Platform",
            description="Production-grade multi-model deployment system",
            version="1.0.0",
            lifespan=lifespan,
        )

        # Health check endpoint
        @app.get("/health")
        async def health_check():
            service_status = {}
            for name, service in self.services.items():
                service_status[name] = getattr(service, 'is_healthy', lambda: True)()

            return {
                "status": "healthy" if all(service_status.values()) else "unhealthy",
                "services": service_status,
                "timestamp": time.time()
            }

        # Model routing endpoint
        @app.post("/route")
        async def route_query(query_data: QueryRequest):
            endpoint = "/route"
            method = "POST"
            start_time = time.perf_counter()
            status_code = 500
            SYSTEM_METRICS.active_requests.labels(endpoint=endpoint).inc()

            try:
                result = await self.services["inference"].process_query(query_data)
                status_code = 502 if result.error else 200
                return JSONResponse(
                    status_code=status_code,
                    content=jsonable_encoder(result),
                )
            except Exception as e:
                status_code = 500
                self.logger.error(f"Query processing error: {e}")
                return JSONResponse(
                    status_code=status_code,
                    content={"error": str(e)},
                )
            finally:
                self._record_api_metrics(
                    endpoint=endpoint,
                    method=method,
                    status_code=status_code,
                    duration_seconds=time.perf_counter() - start_time,
                    error_type="RouteError" if status_code >= 500 else None,
                )
                SYSTEM_METRICS.active_requests.labels(endpoint=endpoint).dec()

        # Metrics endpoint
        @app.get("/metrics")
        async def get_metrics():
            if "monitoring" not in self.services:
                return JSONResponse(
                    status_code=503,
                    content={"error": "Monitoring service is disabled"},
                )
            return await self.services["monitoring"].get_system_metrics()

        return app

    async def _shutdown_services(self):
        """Gracefully shutdown all services"""
        self.logger.info("Shutting down platform services...")

        for name, service in self.services.items():
            try:
                if hasattr(service, 'shutdown'):
                    await service.shutdown()
                self.logger.info(f"Service {name} shutdown successfully")
            except Exception as e:
                self.logger.error(f"Error shutting down service {name}: {e}")

        self.logger.info("Platform shutdown complete")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(self._shutdown_services())
        sys.exit(0)

    async def run(self, include_api: bool = True, include_background: bool = True):
        """Run the selected platform services."""
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        try:
            # Initialize and start services
            await self._initialize_services(
                include_api=include_api,
                include_background=include_background,
            )
            await self._start_services(
                include_api=include_api,
                include_background=include_background,
            )

        except Exception as e:
            self.logger.error(f"Platform startup error: {e}")
            await self._shutdown_services()
            sys.exit(1)


@click.group()
def cli():
    """LLM Router Platform CLI"""
    pass


def create_app() -> FastAPI:
    """Create an API-only FastAPI app for Uvicorn factory mode."""
    platform = LLMRouterPlatform()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await platform._initialize_services(
            include_api=True,
            include_background=False,
        )
        yield
        await platform._shutdown_services()

    return platform._create_fastapi_app(lifespan=lifespan)


@cli.command()
@click.option("--config", default=DEFAULT_CONFIG_PATH, help="Configuration file path")
@click.option("--dev", is_flag=True, help="Run in development mode")
def start(config: str, dev: bool):
    """Start the LLM Router Platform"""
    if dev:
        os.environ[CONFIG_ENV_VAR] = config
        uvicorn.run(
            "main:create_app",
            host="0.0.0.0",
            port=8080,
            factory=True,
            reload=True,
            log_level="debug"
        )
    else:
        platform = LLMRouterPlatform(config_path=config)
        asyncio.run(platform.run(include_api=True, include_background=True))


@cli.command()
@click.option("--config", default=DEFAULT_CONFIG_PATH, help="Configuration file path")
@click.option("--dev", is_flag=True, help="Run API-only development mode")
def start_api(config: str, dev: bool):
    """Start only the API services."""
    os.environ[CONFIG_ENV_VAR] = config
    platform = LLMRouterPlatform(config_path=config)
    api_config = platform.config.get("api", {})
    worker_count = get_api_worker_count(platform.config, dev=dev)

    if dev:
        uvicorn.run(
            "main:create_app",
            host=api_config.get("host", "0.0.0.0"),
            port=api_config.get("port", 8080),
            factory=True,
            reload=True,
            log_level=api_config.get("log_level", "debug"),
        )
        return

    uvicorn.run(
        "main:create_app",
        host=api_config.get("host", "0.0.0.0"),
        port=api_config.get("port", 8080),
        factory=True,
        workers=worker_count,
        log_level=api_config.get("log_level", "info"),
    )


@cli.command()
@click.option("--config", default=DEFAULT_CONFIG_PATH, help="Configuration file path")
def start_workers(config: str):
    """Start only enabled background services."""
    platform = LLMRouterPlatform(config_path=config)
    asyncio.run(platform.run(include_api=False, include_background=True))


@cli.command()
def setup():
    """Setup initial project structure and dependencies"""
    from src.llm_router_part0_setup import setup_project_environment
    setup_project_environment()


@cli.command()
@click.option("--service", help="Specific service to check (router, inference, pipeline, monitoring)")
def health(service: Optional[str]):
    """Check system health"""
    # Implementation for health checks
    click.echo(f"Health check for service: {service or 'all'}")


@cli.command()
@click.option("--output", default="docker-compose.yml", help="Output file for Docker Compose")
def deploy(output: str):
    """Generate deployment configurations"""
    from src.llm_router_part5_deploy import generate_deployment_configs
    generate_deployment_configs(output)


if __name__ == "__main__":
    cli()
