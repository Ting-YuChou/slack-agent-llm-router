"""API key principals and their entitlements (R4)."""

import hashlib
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.api_security import (
    ApiPrincipal,
    api_key_clients_configured,
    apply_principal_entitlements,
    resolve_api_principal,
)
from src.utils.schema import (
    ApiKeyClientConfig,
    ApiKeysSecurityConfig,
    PlatformConfig,
    QueryRequest,
    UserTier,
)


def _sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _api_keys_config(**overrides):
    return ApiKeysSecurityConfig.model_validate(
        {
            "enabled": True,
            "clients": [
                {
                    "client_id": "partner",
                    "key_env": "PARTNER_API_KEY",
                    "max_tier": "premium",
                    "rpm": 120,
                    "burst": 30,
                    "active": 8,
                },
                {
                    "client_id": "ops",
                    "sha256": _sha("ops-secret"),
                    "max_tier": "enterprise",
                    "allow_priority": True,
                },
            ],
            "unmapped_rpm": 30,
            "unmapped_active": 4,
            **overrides,
        }
    ).model_dump(mode="json")


def _env(values):
    return lambda name, default=None: values.get(name, default)


def test_key_env_client_resolves_to_its_principal():
    principal = resolve_api_principal(
        "partner-secret",
        _api_keys_config(),
        anonymous_keys=[],
        getenv=_env({"PARTNER_API_KEY": "partner-secret"}),
    )

    assert principal == ApiPrincipal(
        client_id="partner",
        max_tier="premium",
        rpm=120,
        burst=30,
        active=8,
        allow_priority=False,
        mapped=True,
    )


def test_sha256_client_resolves_without_the_plaintext_key_in_config():
    principal = resolve_api_principal(
        "ops-secret", _api_keys_config(), anonymous_keys=[], getenv=_env({})
    )

    assert principal is not None
    assert principal.client_id == "ops"
    assert principal.max_tier == "enterprise"
    assert principal.allow_priority is True


def test_unmapped_key_becomes_a_free_principal_named_by_its_fingerprint():
    principal = resolve_api_principal(
        "legacy-key",
        _api_keys_config(),
        anonymous_keys=["other-key", "legacy-key"],
        getenv=_env({}),
    )

    assert principal == ApiPrincipal(
        client_id=f"key:{_sha('legacy-key')[:12]}",
        max_tier="free",
        rpm=30,
        burst=30,
        active=4,
        allow_priority=False,
        mapped=False,
    )


@pytest.mark.parametrize("provided", [None, "", "wrong-key"])
def test_unknown_keys_do_not_resolve(provided):
    assert (
        resolve_api_principal(
            provided,
            _api_keys_config(),
            anonymous_keys=["legacy-key"],
            getenv=_env({"PARTNER_API_KEY": "partner-secret"}),
        )
        is None
    )


def test_client_whose_key_env_is_unset_never_matches():
    config = _api_keys_config(
        clients=[{"client_id": "partner", "key_env": "PARTNER_API_KEY"}]
    )

    assert resolve_api_principal("", config, [], getenv=_env({})) is None
    assert not api_key_clients_configured(config, getenv=_env({}))
    assert api_key_clients_configured(
        config, getenv=_env({"PARTNER_API_KEY": "partner-secret"})
    )


def test_entitlements_clamp_tier_and_drop_priority_for_restricted_principals():
    request = QueryRequest(
        query="hi",
        user_id="u1",
        user_tier=UserTier.ENTERPRISE,
        priority=5,
        metadata={
            "requires_low_latency": True,
            "route_to_fast_lane": True,
            "latency_sla": "interactive",
            "fast_lane": True,
            "client_tag": "kept",
        },
    )

    changes = apply_principal_entitlements(
        request, ApiPrincipal(client_id="partner", max_tier="premium")
    )

    assert request.user_tier == UserTier.PREMIUM
    assert request.priority == 1
    assert request.metadata == {"client_tag": "kept"}
    assert changes  # the caller logs what was clamped


def test_entitlements_never_upgrade_the_requested_tier():
    request = QueryRequest(query="hi", user_id="u1", user_tier=UserTier.FREE)

    apply_principal_entitlements(
        request, ApiPrincipal(client_id="ops", max_tier="enterprise")
    )

    assert request.user_tier == UserTier.FREE


def test_allow_priority_principals_keep_priority_and_urgency():
    request = QueryRequest(
        query="hi",
        user_id="u1",
        user_tier=UserTier.ENTERPRISE,
        priority=5,
        metadata={"requires_low_latency": True},
    )

    changes = apply_principal_entitlements(
        request,
        ApiPrincipal(client_id="ops", max_tier="enterprise", allow_priority=True),
    )

    assert request.user_tier == UserTier.ENTERPRISE
    assert request.priority == 5
    assert request.metadata == {"requires_low_latency": True}
    assert changes == []


@pytest.mark.parametrize(
    "entry",
    [
        {"client_id": "partner"},
        {"client_id": "partner", "key_env": "A", "sha256": _sha("a")},
        {"client_id": "partner", "sha256": "not-a-digest"},
        {"client_id": "", "key_env": "A"},
    ],
)
def test_client_entries_need_an_id_and_exactly_one_key_reference(entry):
    with pytest.raises(ValidationError):
        ApiKeyClientConfig.model_validate(entry)


def test_client_ids_must_be_unique():
    with pytest.raises(ValidationError):
        ApiKeysSecurityConfig.model_validate(
            {
                "clients": [
                    {"client_id": "partner", "key_env": "A"},
                    {"client_id": "partner", "key_env": "B"},
                ]
            }
        )


@pytest.mark.parametrize(
    "path",
    ["config/config.yaml", "config/config.compose.yaml", "config/config.demo.yaml"],
)
def test_shipped_configs_bound_unmapped_keys_and_bodies(path):
    with open(Path(__file__).resolve().parents[1] / path, encoding="utf-8") as handle:
        config = PlatformConfig.model_validate(yaml.safe_load(handle))

    assert config.security.api_keys.unmapped_rpm > 0
    assert config.security.api_keys.unmapped_active > 0
    assert config.security.max_request_body_bytes <= 2 * 1024 * 1024


def test_client_defaults_are_free_without_priority():
    client = ApiKeyClientConfig.model_validate({"client_id": "c", "key_env": "K"})

    assert client.max_tier == UserTier.FREE
    assert client.allow_priority is False
    assert client.rpm > 0
    assert client.active > 0
