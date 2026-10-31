"""R16c: routing rule conditions are parsed against a whitelist, not eval'd."""

from pathlib import Path

import pytest
import yaml

from src.llm_router_part1_router import ModelRouter, RoutingRule
from src.routing_rules import ROUTING_RULE_NAMES, RoutingRuleError
from src.utils.schema import ApiConfig, SecurityConfig

ROOT = Path(__file__).resolve().parents[1]
CHECKED_IN_CONFIGS = ("config.yaml", "config.compose.yaml", "config.demo.yaml")


def _rule(condition):
    return RoutingRule(condition=condition, models=["gpt-5"], fallback="gpt-5")


def test_rule_records_the_names_it_reads():
    rule = _rule("query_type == 'analysis' AND token_count > 50000")

    assert rule.referenced_names == frozenset({"query_type", "token_count"})


@pytest.mark.parametrize(
    "condition, context, expected",
    [
        ("user_tier == 'enterprise'", {"user_tier": "enterprise"}, True),
        ("user_tier != 'enterprise'", {"user_tier": "enterprise"}, False),
        ("NOT long_context", {"long_context": False}, True),
        (
            "query_type in ('analysis', 'summarization')",
            {"query_type": "summarization"},
            True,
        ),
        ("query_type not in ['analysis']", {"query_type": "general"}, True),
        ("1000 <= token_count < 5000", {"token_count": 4999}, True),
        ("1000 <= token_count < 5000", {"token_count": 5000}, False),
        (
            "query_type == 'general' OR burst_protection_active == True",
            {"query_type": "analysis", "burst_protection_active": True},
            True,
        ),
        ("cost_sensitivity == None", {"cost_sensitivity": None}, True),
        # Forms the old eval accepted (R16c review).
        (
            "web_search_blocked_reason is None",
            {"web_search_blocked_reason": None},
            True,
        ),
        (
            "web_search_blocked_reason is not None",
            {"web_search_blocked_reason": "x"},
            True,
        ),
        ("None is hint_reason", {"hint_reason": "pin"}, False),
        ("temperature > -1", {"temperature": 0.0}, True),
        ("temperature > -0.5", {"temperature": -1.0}, False),
        (
            "query_type in {'analysis', 'summarization'}",
            {"query_type": "analysis"},
            True,
        ),
        ("query_type not in {'analysis'}", {"query_type": "analysis"}, False),
    ],
)
def test_whitelisted_conditions_evaluate_like_python(condition, context, expected):
    assert _rule(condition).matches(context) is expected


def test_a_known_name_missing_from_the_context_does_not_match():
    rule = _rule("query_type == 'analysis' OR burst_protection_active")

    assert rule.matches({"query_type": "general"}) is False
    assert rule.matches({"query_type": "analysis"}) is True


@pytest.mark.parametrize(
    "condition",
    [
        "__import__('os')",
        "__import__('os').system('true')",
        "query_type.upper() == 'ANALYSIS'",
        "len(query) > 3",
        "allowed_tools[0] == 'web_search'",
        "(lambda: True)()",
        "[name for name in allowed_tools]",
        "query_type == 'a' if True else False",
        "token_count + 1 > 2",
        "token_count > 50*1000",
        "query_type is 'analysis'",
        "query_type is user_tier",
        "-token_count < 0",
        "token_count > -True",
        "-'a' == query_type",
        "query_type in {user_tier}",
        "f'{query_type}' == 'analysis'",
        "query_type := 'analysis'",
        "query_type in (user_tier, 'analysis')",
    ],
)
def test_conditions_outside_the_whitelist_are_rejected(condition):
    with pytest.raises(RoutingRuleError):
        _rule(condition)


@pytest.mark.parametrize("condition", ["query_type ==", "", "query_type = 'a'"])
def test_conditions_that_do_not_parse_are_rejected(condition):
    with pytest.raises(RoutingRuleError):
        _rule(condition)


def test_a_misspelled_name_is_rejected():
    with pytest.raises(RoutingRuleError, match="query_typ"):
        _rule("query_typ == 'analysis'")


def test_a_misspelled_rule_fails_the_router_load(router_config):
    config = dict(router_config)
    config["routing_rules"] = [
        {
            "condition": "user_teir == 'enterprise'",
            "models": ["gpt-5"],
            "fallback": "gpt-5",
        }
    ]

    with pytest.raises(RoutingRuleError, match="user_teir"):
        ModelRouter(config)


@pytest.mark.parametrize(
    "condition, construct",
    [
        ("token_count > 50*1000", "arithmetic"),
        ("len(query) > 3", "function call"),
        ("query_type.upper() == 'A'", "function call"),
        ("query_type.real == 'A'", "attribute access"),
        ("allowed_tools[0] == 'x'", "subscript"),
        ("query_type is 'a'", "'is' and 'is not' compare only with None"),
    ],
)
def test_the_error_names_the_unsupported_construct(condition, construct):
    with pytest.raises(RoutingRuleError, match="not supported") as error:
        _rule(condition)

    assert construct in str(error.value)


def test_a_rule_weight_is_ignored_with_a_deprecation_warning(router_config, caplog):
    config = dict(router_config)
    config["routing_rules"] = [
        {
            "condition": "query_type == 'analysis'",
            "models": ["gpt-5"],
            "fallback": "gpt-5",
            "weight": 2.0,
        }
    ]

    with caplog.at_level("WARNING", logger="src.llm_router_part1_router"):
        router = ModelRouter(config)

    assert [rule.condition for rule in router.routing_rules] == [
        "query_type == 'analysis'"
    ]
    assert any(
        "weight" in record.getMessage() and "deprecated" in record.getMessage()
        for record in caplog.records
    )


def test_a_malformed_rule_entry_fails_the_router_load(router_config):
    config = dict(router_config)
    config["routing_rules"] = [{"condition": "query_type == 'analysis'"}]

    with pytest.raises(RoutingRuleError):
        ModelRouter(config)


@pytest.mark.parametrize("filename", CHECKED_IN_CONFIGS)
def test_checked_in_rules_only_read_known_names(filename):
    payload = yaml.safe_load((ROOT / "config" / filename).read_text(encoding="utf-8"))

    for rule_config in payload["router"].get("routing_rules") or []:
        rule = RoutingRule(**rule_config)
        assert rule.referenced_names
        assert rule.referenced_names <= ROUTING_RULE_NAMES


@pytest.mark.asyncio
async def test_every_rule_name_is_in_the_routing_context(router_config):
    from src.utils.schema import QueryRequest, UserTier

    class _PolicyStub:
        async def get_effective_policy(self, request_id, user_id, session_id):
            return {"policy_source": "request"}

    router = ModelRouter(router_config, policy_cache=_PolicyStub())
    router.token_counter.count_tokens = lambda _text, _model="default": 128

    context = await router._build_query_context(
        QueryRequest(query="Hello", user_id="u1", user_tier=UserTier.PREMIUM)
    )

    assert ROUTING_RULE_NAMES <= set(context)


def test_dead_router_state_and_config_are_gone(router_config):
    router = ModelRouter(router_config)

    assert not hasattr(router, "request_cache")
    assert "weight" not in RoutingRule.__dataclass_fields__
    assert "cors_origins" not in ApiConfig.model_fields
    assert "jwt" not in SecurityConfig.model_fields


@pytest.mark.parametrize("filename", CHECKED_IN_CONFIGS)
def test_checked_in_configs_drop_the_dead_keys(filename):
    payload = yaml.safe_load((ROOT / "config" / filename).read_text(encoding="utf-8"))

    assert "cors_origins" not in (payload.get("api") or {})
    assert "jwt" not in (payload.get("security") or {})


def test_the_unused_model_engine_module_is_gone():
    assert not (ROOT / "src" / "models" / "mistral_engine.py").exists()
