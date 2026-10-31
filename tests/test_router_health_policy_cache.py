"""O4: an unhealthy policy cache never makes the router unready."""

from types import SimpleNamespace

from src.llm_router_part1_router import ModelRouter


def _ready_router(router_config, policy_cache):
    router = ModelRouter(router_config, policy_cache=policy_cache)
    router._initialized = True
    router.classifier._is_initialized = True
    return router


def test_unhealthy_policy_cache_does_not_unready_the_router(router_config):
    cache = SimpleNamespace(is_healthy=lambda: False)

    assert _ready_router(router_config, cache).is_healthy() is True


def test_router_health_does_not_ask_the_policy_cache(router_config):
    def explode():
        raise AssertionError("router health must not read the policy cache")

    cache = SimpleNamespace(is_healthy=explode)

    assert _ready_router(router_config, cache).is_healthy() is True


def test_router_without_its_default_model_is_still_unready(router_config):
    router = _ready_router(router_config, SimpleNamespace(is_healthy=lambda: True))
    router.default_model = "missing-model"

    assert router.is_healthy() is False
