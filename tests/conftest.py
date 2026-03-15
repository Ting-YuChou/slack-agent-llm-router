import importlib.util
import importlib.machinery
import sys
import types
from datetime import datetime

import pytest


def _ensure_module(name: str):
    if name in sys.modules:
        return sys.modules[name]

    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    sys.modules[name] = module

    if "." in name:
        parent_name, child_name = name.rsplit(".", 1)
        parent = _ensure_module(parent_name)
        setattr(parent, child_name, module)

    return module


def _missing(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is None


if True:
    tiktoken = _ensure_module("tiktoken")

    class _Encoder:
        def encode(self, text):
            return text.split()

    tiktoken.encoding_for_model = lambda _name: _Encoder()
    tiktoken.get_encoding = lambda _name: _Encoder()


if True:
    transformers = _ensure_module("transformers")

    class _Tokenizer:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def encode(self, text):
            return text.split()

    transformers.AutoTokenizer = _Tokenizer
    transformers.AutoModel = object
    transformers.AutoModelForCausalLM = object
    transformers.GenerationConfig = object
    transformers.PreTrainedTokenizer = object
    transformers.PreTrainedModel = object


if True:
    torch = _ensure_module("torch")
    torch.nn = _ensure_module("torch.nn")
    torch.nn.functional = _ensure_module("torch.nn.functional")


if _missing("numpy"):
    numpy = _ensure_module("numpy")
    numpy.array = lambda value: value


if True:
    sentence_transformers = _ensure_module("sentence_transformers")

    class _SentenceTransformer:
        def __init__(self, *_args, **_kwargs):
            pass

    sentence_transformers.SentenceTransformer = _SentenceTransformer


if True:
    sklearn = _ensure_module("sklearn")
    feature_extraction = _ensure_module("sklearn.feature_extraction")
    text = _ensure_module("sklearn.feature_extraction.text")
    metrics = _ensure_module("sklearn.metrics")
    pairwise = _ensure_module("sklearn.metrics.pairwise")

    class _TfidfVectorizer:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    text.TfidfVectorizer = _TfidfVectorizer
    pairwise.cosine_similarity = lambda *_args, **_kwargs: [[1.0]]
    feature_extraction.text = text
    metrics.pairwise = pairwise
    sklearn.feature_extraction = feature_extraction
    sklearn.metrics = metrics


if _missing("openai"):
    openai = _ensure_module("openai")
    openai.AsyncOpenAI = object


if _missing("anthropic"):
    anthropic = _ensure_module("anthropic")
    anthropic.AsyncAnthropic = object


if _missing("redis"):
    redis = _ensure_module("redis")
    redis_asyncio = _ensure_module("redis.asyncio")

    class _Redis:
        def __init__(self, *args, **kwargs):
            self.store = {}

        async def ping(self):
            return True

        async def get(self, key):
            return self.store.get(key)

        async def setex(self, key, _ttl, value):
            self.store[key] = value

    redis_asyncio.Redis = _Redis
    redis.asyncio = redis_asyncio


if _missing("aiokafka"):
    aiokafka = _ensure_module("aiokafka")
    aiokafka_errors = _ensure_module("aiokafka.errors")

    class _Producer:
        def __init__(self, *args, **kwargs):
            pass

        async def start(self):
            return None

        async def stop(self):
            return None

        async def send(self, *args, **kwargs):
            return None

    class _Consumer:
        def __init__(self, *args, **kwargs):
            pass

        async def start(self):
            return None

        async def stop(self):
            return None

        def __aiter__(self):
            async def _empty():
                if False:
                    yield None

            return _empty()

    aiokafka.AIOKafkaProducer = _Producer
    aiokafka.AIOKafkaConsumer = _Consumer
    aiokafka_errors.KafkaError = Exception


if _missing("clickhouse_connect"):
    clickhouse_connect = _ensure_module("clickhouse_connect")
    clickhouse_driver = _ensure_module("clickhouse_connect.driver")

    class _Client:
        def query(self, *_args, **_kwargs):
            return types.SimpleNamespace(result_rows=[[1]])

        def command(self, *_args, **_kwargs):
            return None

        def insert(self, *_args, **_kwargs):
            return None

        def close(self):
            return None

    clickhouse_connect.get_client = lambda **_kwargs: _Client()
    clickhouse_driver.Client = _Client


if _missing("GPUtil"):
    gputil = _ensure_module("GPUtil")
    gputil.getGPUs = lambda: []


if _missing("psutil"):
    psutil = _ensure_module("psutil")
    psutil.cpu_percent = lambda interval=None: 0.0
    psutil.virtual_memory = lambda: types.SimpleNamespace(percent=0.0)
    psutil.disk_usage = lambda _path: types.SimpleNamespace(used=0, total=1)
    psutil.net_io_counters = lambda: types.SimpleNamespace(bytes_sent=0, bytes_recv=0)
    psutil.pids = lambda: []
    psutil.boot_time = lambda: 0


if _missing("httpx"):
    httpx = _ensure_module("httpx")

    class _AsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, *_args, **_kwargs):
            return types.SimpleNamespace(status_code=200, json=lambda: {})

        async def post(self, *_args, **_kwargs):
            return types.SimpleNamespace(
                status_code=200,
                json=lambda: {},
                raise_for_status=lambda: None,
            )

    httpx.AsyncClient = _AsyncClient


slack_sdk = _ensure_module("slack_sdk")
web = _ensure_module("slack_sdk.web")
async_client = _ensure_module("slack_sdk.web.async_client")
socket_mode = _ensure_module("slack_sdk.socket_mode")
socket_mode_async = _ensure_module("slack_sdk.socket_mode.async_client")
socket_mode_request = _ensure_module("slack_sdk.socket_mode.request")
socket_mode_response = _ensure_module("slack_sdk.socket_mode.response")


class _AsyncWebClient:
    def __init__(self, *args, **kwargs):
        pass


class _AsyncSocketModeClient:
    def __init__(self, *args, **kwargs):
        self.socket_mode_request_listeners = []


class _SocketModeRequest:
    def __init__(self, *args, **kwargs):
        self.type = ""
        self.payload = {}
        self.envelope_id = "env"


class _SocketModeResponse:
    def __init__(self, envelope_id):
        self.envelope_id = envelope_id


async_client.AsyncWebClient = _AsyncWebClient
socket_mode_async.AsyncSocketModeClient = _AsyncSocketModeClient
socket_mode_request.SocketModeRequest = _SocketModeRequest
socket_mode_response.SocketModeResponse = _SocketModeResponse
slack_sdk.web = web
slack_sdk.socket_mode = socket_mode


if _missing("prometheus_client"):
    prometheus_client = _ensure_module("prometheus_client")
    prometheus_metrics = _ensure_module("prometheus_client.metrics")

    class _Metric:
        def __init__(self, *args, **kwargs):
            self._current = 0
            self._children = {}

        def labels(self, *args, **kwargs):
            key = args or tuple(sorted(kwargs.items()))
            child = self._children.get(key)
            if child is None:
                child = _Metric()
                self._children[key] = child
            return child

        def inc(self, amount=1):
            self._current += amount

        def observe(self, amount):
            self._current = amount

        def set(self, amount):
            self._current = amount

        @property
        def _value(self):
            return types.SimpleNamespace(get=lambda: self._current)

    prometheus_client.Counter = _Metric
    prometheus_client.Gauge = _Metric
    prometheus_client.Histogram = _Metric
    prometheus_client.Summary = _Metric
    prometheus_client.Info = _Metric
    prometheus_client.Enum = _Metric
    prometheus_client.CollectorRegistry = object
    prometheus_client.generate_latest = lambda *_args, **_kwargs: b""
    prometheus_client.CONTENT_TYPE_LATEST = "text/plain"
    prometheus_metrics.MetricWrapperBase = _Metric


if _missing("tenacity"):
    tenacity = _ensure_module("tenacity")

    def _retry(*_args, **_kwargs):
        def _decorator(func):
            return func

        return _decorator

    tenacity.retry = _retry
    tenacity.stop_after_attempt = lambda *_args, **_kwargs: None
    tenacity.wait_exponential = lambda *_args, **_kwargs: None


if _missing("structlog"):
    structlog = _ensure_module("structlog")
    structlog.configure = lambda *args, **kwargs: None
    structlog.get_logger = lambda *args, **kwargs: types.SimpleNamespace(
        bind=lambda **_kw: None,
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
        debug=lambda *a, **k: None,
    )


if _missing("pythonjsonlogger"):
    pythonjsonlogger = _ensure_module("pythonjsonlogger")
    jsonlogger = _ensure_module("pythonjsonlogger.jsonlogger")

    class _JsonFormatter:
        def __init__(self, *args, **kwargs):
            pass

        def add_fields(self, log_record, record, message_dict):
            if message_dict:
                log_record.update(message_dict)

    jsonlogger.JsonFormatter = _JsonFormatter
    pythonjsonlogger.jsonlogger = jsonlogger


@pytest.fixture
def sample_query_request():
    from src.utils.schema import QueryRequest, UserTier

    return QueryRequest(
        query="Write a Python function to add two numbers",
        user_id="user-123",
        user_tier=UserTier.PREMIUM,
        context="Previous discussion context",
        max_tokens=256,
        temperature=0.3,
    )


@pytest.fixture
def inference_response_factory():
    from src.utils.schema import InferenceResponse

    def _build(**overrides):
        payload = {
            "response_text": "generated response",
            "model_name": "gpt-5",
            "provider": "openai",
            "token_count_input": 10,
            "token_count_output": 20,
            "total_tokens": 30,
            "latency_ms": 120,
            "tokens_per_second": 166.67,
            "cost_usd": 0.012,
            "cached": False,
        }
        payload.update(overrides)
        return InferenceResponse(**payload)

    return _build


@pytest.fixture
def router_config():
    return {
        "default_model": "mistral-7b",
        "routing_strategy": "intelligent",
        "models": {
            "gpt-5": {
                "provider": "openai",
                "max_tokens": 8192,
                "cost_per_token": 0.00003,
                "priority": 1,
                "capabilities": ["reasoning", "coding", "analysis"],
                "api_key_env": "OPENAI_API_KEY",
            },
            "mistral-7b": {
                "provider": "vllm",
                "model_path": "/models/mistral",
                "max_tokens": 4096,
                "cost_per_token": 0.0,
                "priority": 3,
                "capabilities": ["general", "coding"],
                "gpu_memory_gb": 16,
            },
        },
        "routing_rules": [
            {
                "condition": "query_type == 'code_generation'",
                "models": ["gpt-5", "mistral-7b"],
                "fallback": "mistral-7b",
            }
        ],
    }


@pytest.fixture
def fixed_datetime():
    return datetime(2026, 3, 12, 10, 30, 0)
