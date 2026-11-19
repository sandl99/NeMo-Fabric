# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Discord configuration across startup, retained state, and reloads."""

import json
import os
import subprocess
import sys
from unittest.mock import AsyncMock

import pytest
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapters.hermes import configuration
from nemo_fabric_adapters.hermes.adapter import HermesRuntime

pytestmark = pytest.mark.usefixtures("requires_hermes_agent")


PROBE = r"""
import json
import os
import sys
from types import ModuleType
from nemo_fabric_adapters.hermes import native

def observe():
    from gateway.config import load_gateway_config, Platform
    from plugins.platforms.discord.adapter import DiscordAdapter
    config = load_gateway_config().platforms[Platform.DISCORD]
    adapter = DiscordAdapter(config)
    adapter._snapshot_gate_env()
    adapter._allowed_user_ids = adapter._get_allowed_users()
    adapter._allowed_role_ids = adapter._get_allowed_roles()
    return {
        "selected_token": config.token == "selected-fixture-discord-credential",
        "users": sorted(adapter._get_allowed_users()),
        "channels": sorted(adapter._get_allowed_channels()),
        "mention": adapter._discord_require_mention(),
        "roles": sorted(adapter._get_allowed_roles()),
        "outsider_allowed": adapter._is_allowed_user("999999999999999999", is_dm=True),
    }

def probe():
    from hermes_cli.config import reload_env
    from hermes_cli.env_loader import load_hermes_dotenv
    from agent.secret_scope import build_profile_secret_scope, set_secret_scope, reset_secret_scope
    from pathlib import Path
    results = {"startup": observe()}
    load_hermes_dotenv()
    results["dotenv_reload"] = observe()
    reload_env()
    results["explicit_reload"] = observe()
    if not os.environ.get("QA_NATIVE_BASELINE"):
        from nemo_fabric_adapter_contract.models import AgentConfig
        settings = json.loads((Path(os.environ["HERMES_HOME"]) / "fabric-config.json").read_text())
        native.configure_discord_environment(AgentConfig.from_mapping(settings))
    load_hermes_dotenv()
    reload_env()
    results["repeated_setup"] = observe()
    scope = build_profile_secret_scope(hermes_home=Path(os.environ["HERMES_HOME"]))
    token = set_secret_scope(scope)
    try:
        results["rebuilt_scope"] = observe()
    finally:
        reset_secret_scope(token)
    foreign = Path(os.environ["HERMES_HOME"]) / "foreign"
    foreign.mkdir(exist_ok=True)
    (foreign / ".env").write_text("DISCORD_BOT_TOKEN=foreign-fixture-credential\nDISCORD_ALLOWED_USERS=777777777777777777\n")
    other = build_profile_secret_scope(hermes_home=foreign)
    print(json.dumps({"phases": results, "foreign_scope_preserved": other.get("DISCORD_BOT_TOKEN") == "foreign-fixture-credential" and other.get("DISCORD_ALLOWED_USERS") == "777777777777777777"}))

cli = ModuleType("hermes_cli.main")
cli.main = probe
sys.modules["hermes_cli.main"] = cli
if os.environ.get("QA_NATIVE_BASELINE"):
    probe()
else:
    native.main()
"""


@pytest.fixture(name="discord_native")
def discord_native_fixture(tmp_path):
    pytest.importorskip("discord")
    home = tmp_path / "hermes"
    home.mkdir()
    return home


def _agent(discord):
    return AgentConfig.from_mapping(
        {
            "harness": {
                "settings": {
                    "sdk": {"enabled": False},
                    "platforms": {"discord": discord},
                }
            },
            "models": {"default": {"provider": "nvidia", "model": "fixture-model"}},
        }
    )


def _native_observations(home, *, source, authored, baseline=False):
    agent = _agent(authored)
    configuration.write_hermes_config(agent, home, workspace=".")
    snapshot = home / "fabric-config.json"
    snapshot.write_text(json.dumps(agent.to_mapping()))
    environment = {
        **os.environ,
        "HERMES_HOME": str(home),
        "DISCORD_BOT_TOKEN": "ambient-default-discord-credential",
        "SELECTED_DISCORD_TOKEN": "selected-fixture-discord-credential",
    }
    if authored.get("token_env", "DISCORD_BOT_TOKEN") == "DISCORD_BOT_TOKEN":
        environment["DISCORD_BOT_TOKEN"] = "selected-fixture-discord-credential"
    conflicts = {
        "DISCORD_ALLOWED_USERS": "999999999999999999",
        "DISCORD_ALLOWED_CHANNELS": "888888888888888888",
        "DISCORD_ALLOWED_ROLES": "555555555555555555",
        "DISCORD_REQUIRE_MENTION": "false",
        "DISCORD_ALLOW_ALL_USERS": "true",
        "GATEWAY_ALLOW_ALL_USERS": "true",
        "GATEWAY_ALLOWED_USERS": "*",
    }
    if baseline:
        environment["QA_NATIVE_BASELINE"] = "true"
    if source == "ambient":
        environment.update(conflicts)
    else:
        (home / ".env").write_text(
            "DISCORD_BOT_TOKEN=retained-default-discord-credential\n"
            "SELECTED_DISCORD_TOKEN=retained-selected-discord-credential\n"
            + "".join(f"{key}={value}\n" for key, value in conflicts.items())
        )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            PROBE,
            "--fabric-config",
            str(snapshot),
            "gateway",
            "run",
        ],
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )
    assert (
        "selected-fixture-discord-credential" not in (home / "config.yaml").read_text()
    )
    assert "selected-fixture-discord-credential" not in snapshot.read_text()
    observations = json.loads(result.stdout.splitlines()[-1])
    assert observations["foreign_scope_preserved"] is True
    return observations["phases"]


@pytest.mark.parametrize("source", ["ambient", "retained"])
@pytest.mark.parametrize("token_env", ["DISCORD_BOT_TOKEN", "SELECTED_DISCORD_TOKEN"])
def test_native_gateway_keeps_the_selected_bot_credential(
    discord_native, source, token_env
):
    observations = _native_observations(
        discord_native,
        source=source,
        authored={"enabled": True, "token_env": token_env},
    )

    assert all(value["selected_token"] for value in observations.values()), observations


@pytest.mark.parametrize("source", ["ambient", "retained"])
def test_native_gateway_keeps_selected_access_settings_across_reload(
    discord_native, source
):
    observations = _native_observations(
        discord_native,
        source=source,
        authored={
            "enabled": True,
            "token_env": "SELECTED_DISCORD_TOKEN",
            "allowed_users": ["123456789012345678"],
            "allowed_channels": ["234567890123456789"],
            "require_mention": True,
        },
    )

    for value in observations.values():
        assert value["users"] == ["123456789012345678"]
        assert value["channels"] == ["234567890123456789"]
        assert value["mention"] is True
        assert value["roles"] == []
        assert value["outsider_allowed"] is False


def test_native_gateway_retains_native_policy_for_omitted_fields(discord_native):
    observations = _native_observations(
        discord_native, source="ambient", authored={"enabled": True}
    )

    baseline = _native_observations(
        discord_native, source="ambient", authored={"enabled": True}, baseline=True
    )
    for phase, value in observations.items():
        assert {
            key: item for key, item in value.items() if key != "selected_token"
        } == {
            key: item
            for key, item in baseline[phase].items()
            if key != "selected_token"
        }


@pytest.mark.usefixtures("discord_native")
async def test_discord_only_keeps_api_disabled_with_retained_credentials(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "selected-fixture-discord-credential")
    monkeypatch.setenv("NVIDIA_API_KEY", "fixture-model-key")
    runtime = HermesRuntime()
    observed = {}

    async def inspect_gateway(*args, hermes_home, **kwargs):
        # Reuse Hermes's real .env reload and config loader in an isolated child.
        (hermes_home / ".env").write_text(
            "API_SERVER_KEY=ZQRn7YmCPVqN4dJt83jKuGb9WxR2avLc\n"
            "API_SERVER_HOST=127.0.0.1\nAPI_SERVER_PORT=18888\n"
        )
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from hermes_cli.config import reload_env; reload_env(); "
                "from gateway.config import load_gateway_config, Platform; "
                "api=load_gateway_config().platforms.get(Platform.API_SERVER); "
                "print(bool(api and api.enabled))",
            ],
            env={**os.environ, "HERMES_HOME": str(hermes_home)},
            text=True,
            capture_output=True,
            timeout=30,
            check=True,
        )
        observed["api_enabled"] = result.stdout.strip() == "True"

    runtime._interfaces.start = AsyncMock(side_effect=inspect_gateway)
    try:
        await runtime.start(
            {
                "base_dir": str(tmp_path),
                "config": _agent({"enabled": True}),
                "runtime_context": {
                    "runtime_id": "discord-native",
                    "invocation_id": "start",
                    "request_id": "start",
                    "artifacts": {},
                    "environment": {
                        "environment_id": "local",
                        "provider": "local",
                        "control_location": "in_env_control",
                        "ownership": "caller_owned",
                        "workspace": str(tmp_path),
                    },
                },
            }
        )
        assert observed["api_enabled"] is False
    finally:
        await runtime.stop()
