"""S17d: Socket Mode never verifies request signatures, so no signing secret."""

from pathlib import Path

import pytest
import yaml

from src.utils.schema import SlackConfig

ROOT = Path(__file__).resolve().parents[1]


def test_slack_config_has_no_signing_secret_fields():
    assert "signing_secret" not in SlackConfig.model_fields
    assert "signing_secret_env" not in SlackConfig.model_fields


@pytest.mark.parametrize(
    "path",
    ["config/config.yaml", "config/config.compose.yaml", "config/config.demo.yaml"],
)
def test_shipped_configs_set_no_signing_secret(path):
    slack = yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))["slack"]

    assert "signing_secret" not in slack
    assert "signing_secret_env" not in slack
    # The keys the bot does read are still there.
    assert slack["bot_token_env"] == "SLACK_BOT_TOKEN"
    assert slack["app_token_env"] == "SLACK_APP_TOKEN"


def test_an_old_config_with_a_signing_secret_still_loads():
    config = SlackConfig(signing_secret_env="SLACK_SIGNING_SECRET")

    assert config.bot_token_env is None
