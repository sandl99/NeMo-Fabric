# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discord configuration admission and Hermes-native configuration mapping."""

import asyncio
import importlib.util
import sys
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
import yaml
from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric.errors import FabricConfigError
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapters.common.lifecycle import LifecycleError
from nemo_fabric_adapters.hermes import configuration
from nemo_fabric_adapters.hermes import interfaces
from nemo_fabric_adapters.hermes.adapter import HermesRuntime


def _config(platforms):
    return {
        "harness": {
            "settings": {"platforms": platforms},
        },
        "models": {"default": {"provider": "nvidia", "model": "test-model"}},
    }


def _fabric_config(platforms):
    raw = _config(platforms)
    raw["harness"]["adapter_id"] = "nvidia.fabric.hermes"
    return FabricConfig.from_mapping({"metadata": {"name": "discord-config"}, **raw})


@pytest.mark.parametrize("enabled", [False, True])
def test_plan_accepts_discord_configuration_without_bot_credentials(tmp_path, enabled):
    config = _fabric_config(
        {
            "discord": {
                "enabled": enabled,
                "token_env": "FABRIC_DISCORD_TOKEN",
                "allowed_users": ["123456789012345678"],
                "allowed_channels": ["234567890123456789"],
                "require_mention": True,
            }
        }
    )

    Fabric().plan(config, base_dir=tmp_path)


@pytest.mark.parametrize(
    "platforms",
    [
        {"signal": {"enabled": False}},
        {"discord": {"token": "inline-secret"}},
        {"discord": {"token_env": "not-an-env-name"}},
        {"discord": {"token_env": ""}},
        {"discord": {"enabled": "true"}},
        {"discord": {"allowed_users": [123]}},
        {"discord": {"allowed_users": ["*"]}},
        {"discord": {"allowed_channels": ["not-a-channel-id"]}},
        {"discord": {"require_mention": "false"}},
        {"discord": {"unknown": True}},
    ],
)
def test_plan_rejects_unsupported_or_unsafe_discord_settings(tmp_path, platforms):
    with pytest.raises(FabricConfigError):
        Fabric().plan(_fabric_config(platforms), base_dir=tmp_path)


@pytest.mark.parametrize("enabled", [False, True])
def test_discord_settings_reach_native_config_with_only_a_secret_reference(
    tmp_path, monkeypatch, enabled
):
    monkeypatch.setenv("FABRIC_DISCORD_TOKEN", "private-discord-token")
    discord = {
        "enabled": enabled,
        "token_env": "FABRIC_DISCORD_TOKEN",
        "allowed_users": ["123456789012345678"],
        "allowed_channels": ["234567890123456789"],
        "require_mention": False,
    }
    agent = AgentConfig.from_mapping(_config({"discord": discord}))

    path, _ = configuration.write_hermes_config(agent, tmp_path, workspace=".")
    native = yaml.safe_load(path.read_text())

    assert native["platforms"]["discord"] == {
        "enabled": enabled,
        "token": "${FABRIC_DISCORD_TOKEN}",
        "extra": {
            "allowed_users": ["123456789012345678"],
            "allowed_channels": ["234567890123456789"],
            "require_mention": False,
        },
    }
    assert "private-discord-token" not in path.read_text()
    assert discord["token_env"] == "FABRIC_DISCORD_TOKEN"


def test_discord_configuration_defaults_to_disabled_without_granting_senders():
    config = configuration.build_hermes_config(
        AgentConfig.from_mapping(_config({"discord": {}})), workspace="."
    )

    assert config["platforms"]["discord"] == {"enabled": False}


def test_omitted_discord_configuration_preserves_existing_native_settings():
    config = configuration.build_hermes_config(
        AgentConfig.from_mapping(_config({})), workspace="."
    )

    assert "platforms" not in config


@pytest.mark.parametrize("api", [False, True])
async def test_missing_discord_credential_fails_before_creating_native_state(
    tmp_path, monkeypatch, api
):
    monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
    raw = _config({"discord": {"enabled": True}})
    if api:
        raw["harness"]["settings"]["interfaces"] = {"api": {"port": 8642}}
    runtime = HermesRuntime()
    payload = {
        "base_dir": str(tmp_path),
        "config": AgentConfig.from_mapping(raw),
        "runtime_context": {
            "runtime_id": "discord-start",
            "invocation_id": "invocation",
            "request_id": "request",
            "environment": {
                "environment_id": "local",
                "provider": "local",
                "control_location": "in_env_control",
                "ownership": "caller_owned",
                "workspace": str(tmp_path),
            },
            "artifacts": {},
        },
    }

    with pytest.raises(LifecycleError) as caught:
        await runtime.start(payload)

    assert caught.value.code == "hermes_discord_credential_missing"
    assert not (tmp_path / "artifacts").exists()


def test_discord_is_an_execution_path_without_sdk_or_api(tmp_path):
    config = _fabric_config({"discord": {"enabled": True}})
    config.harness.settings["sdk"] = {"enabled": False}

    Fabric().plan(config, base_dir=tmp_path)


@pytest.mark.parametrize(
    "platforms", [{}, {"discord": {}}, {"discord": {"enabled": False}}]
)
def test_disabled_discord_does_not_create_an_execution_path(tmp_path, platforms):
    config = _fabric_config(platforms)
    config.harness.settings["sdk"] = {"enabled": False}

    with pytest.raises(FabricConfigError):
        Fabric().plan(config, base_dir=tmp_path)


@pytest.fixture(name="managed_gateway")
def managed_gateway_fixture(tmp_path, monkeypatch):
    process = MagicMock(pid=12345, returncode=None)

    async def wait():
        process.returncode = 0
        return 0

    process.wait = AsyncMock(side_effect=wait)
    spawn = AsyncMock(return_value=process)
    monkeypatch.setattr(interfaces.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(interfaces, "_http_status", AsyncMock(return_value=200))
    monkeypatch.setattr(
        interfaces.local_interfaces, "port_available", lambda port: True
    )
    signal = MagicMock()
    monkeypatch.setattr(interfaces.common_utils, "signal_process_group", signal)
    status = MagicMock()
    status.read_runtime_status.return_value = {
        "pid": process.pid,
        "start_time": 100,
        "gateway_state": "running",
        "platforms": {
            "discord": {
                "state": "connected",
                "writer_pid": process.pid,
                "writer_start_time": 100,
            }
        },
    }
    status.runtime_status_is_stale.return_value = False
    status.runtime_status_pid_is_live.return_value = True
    status.start_time_fingerprints_match.return_value = True
    monkeypatch.setitem(sys.modules, "gateway", MagicMock())
    monkeypatch.setitem(sys.modules, "gateway.status", status)
    return {
        "owner": interfaces.InterfaceChildren(),
        "process": process,
        "spawn": spawn,
        "signal": signal,
        "status": status,
        "home": tmp_path,
        "token_path": tmp_path / "interfaces/api.token",
    }


async def _start_gateway(managed, *, api=False, timeout=1):
    agent = AgentConfig.from_mapping(_config({"discord": {"enabled": True}}))
    await managed["owner"].start(
        {"api": 8642} if api else {},
        hermes_home=managed["home"],
        token_path=managed["token_path"],
        timeout=timeout,
        environment={"API_SERVER_KEY": "ambient-key", "API_SERVER_ENABLED": "true"},
        agent_config=agent,
    )


@pytest.mark.parametrize("api", [False, True])
async def test_discord_reuses_one_native_gateway_and_its_cleanup(managed_gateway, api):
    managed = managed_gateway
    try:
        await _start_gateway(managed, api=api)

        assert managed["spawn"].await_count == 1
        assert managed["spawn"].call_args.args[-2:] == ("gateway", "run")
        environment = managed["spawn"].call_args.kwargs["env"]
        assert ("API_SERVER_KEY" in environment) is api
        assert managed["token_path"].exists() is api
        assert await managed["owner"].check("ready") is None
    finally:
        await managed["owner"].stop()

    assert managed["signal"].called
    assert not managed["token_path"].exists()
    assert not (managed["home"] / "interfaces/agent-config.json").exists()


@pytest.mark.parametrize("state", ["connecting", "retrying", "disconnected", "fatal"])
async def test_discord_connection_failure_is_visible_in_gateway_health(
    managed_gateway, state
):
    managed = managed_gateway
    try:
        await _start_gateway(managed, api=True)
        managed["status"].read_runtime_status.return_value["platforms"]["discord"][
            "state"
        ] = state

        assert await managed["owner"].check("ready") == "endpoint_unavailable"
    finally:
        await managed["owner"].stop()


@pytest.mark.parametrize(
    "observation",
    [
        "missing",
        "foreign-pid",
        "stale",
        "reused-pid",
        "foreign-channel-writer",
        "retained-channel",
    ],
)
async def test_discord_readiness_rejects_unconfirmed_native_status(
    managed_gateway, observation
):
    managed = managed_gateway
    try:
        await _start_gateway(managed)
        if observation == "missing":
            managed["status"].read_runtime_status.return_value = None
        elif observation == "foreign-pid":
            managed["status"].read_runtime_status.return_value["pid"] = 99999
        elif observation == "stale":
            managed["status"].runtime_status_is_stale.return_value = True
        elif observation == "reused-pid":
            managed["status"].runtime_status_pid_is_live.return_value = False
        elif observation == "foreign-channel-writer":
            managed["status"].read_runtime_status.return_value["platforms"]["discord"][
                "writer_pid"
            ] = 99999
        else:
            managed["status"].start_time_fingerprints_match.return_value = False

        assert await managed["owner"].check("ready") == "endpoint_unavailable"
    finally:
        await managed["owner"].stop()


@pytest.mark.parametrize("api", [False, True])
async def test_discord_startup_timeout_stops_the_owned_gateway(managed_gateway, api):
    managed = managed_gateway
    managed["status"].read_runtime_status.return_value = None

    with pytest.raises(LifecycleError) as caught:
        await _start_gateway(managed, api=api, timeout=0.01)

    assert caught.value.code == "hermes_interface_start_timeout"
    assert managed["process"].returncode == 0
    assert not (managed["home"] / "interfaces/agent-config.json").exists()
    assert not managed["token_path"].exists()


async def test_readiness_cannot_succeed_after_the_startup_deadline(
    managed_gateway, monkeypatch
):
    async def slow_health(*args):
        await asyncio.sleep(0.1)
        return 200

    monkeypatch.setattr(interfaces, "_http_status", slow_health)
    with pytest.raises(LifecycleError) as caught:
        await _start_gateway(managed_gateway, api=True, timeout=0.01)

    assert caught.value.code == "hermes_interface_start_timeout"
    assert managed_gateway["process"].returncode == 0


def test_discord_only_gateway_uses_the_native_model_route_and_bounded_drain():
    raw = _config({"discord": {"enabled": True}})
    raw["models"]["default"] = {
        "provider": "openai",
        "model": "configured-model",
        "base_url": "http://inference.example/v1",
        "api_key_env": "MODEL_ROUTE_KEY",
    }

    native = configuration.build_hermes_config(
        AgentConfig.from_mapping(raw), workspace="."
    )

    assert native["model"]["provider"] == "custom"
    assert native["model"]["base_url"] == "http://inference.example/v1"
    assert native["model"]["api_key"] == "${MODEL_ROUTE_KEY}"
    assert (
        native["agent"]["cron_drain_timeout"]
        == interfaces.GATEWAY_DRAIN_TIMEOUT_SECONDS
    )


@pytest.mark.parametrize("sdk", [True, False])
async def test_missing_discord_library_fails_before_native_state(
    tmp_path, monkeypatch, sdk
):
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "private-bot-token")
    original = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: None if name == "discord" else original(name),
    )
    raw = _config({"discord": {"enabled": True}})
    raw["harness"]["settings"]["sdk"] = {"enabled": sdk}
    runtime = HermesRuntime()

    with pytest.raises(LifecycleError) as caught:
        await runtime.start(
            {
                "base_dir": str(tmp_path),
                "config": AgentConfig.from_mapping(raw),
                "runtime_context": {
                    "runtime_id": "missing-discord",
                    "invocation_id": "invocation",
                    "request_id": "request",
                    "environment": {
                        "environment_id": "local",
                        "provider": "local",
                        "control_location": "in_env_control",
                        "ownership": "caller_owned",
                        "workspace": str(tmp_path),
                    },
                    "artifacts": {},
                },
            }
        )

    assert caught.value.code == "hermes_discord_dependency_missing"
    assert "private-bot-token" not in str(caught.value)
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize("channel", ["telegram", "slack", "weixin"])
@pytest.mark.parametrize("api", [False, True])
async def test_selected_channels_share_gateway_and_require_connection(
    managed_gateway, channel, api
):
    managed = managed_gateway
    agent = AgentConfig.from_mapping(_config({channel: {"enabled": True}}))
    managed["status"].read_runtime_status.return_value["platforms"][channel] = {
        "state": "connected",
        "writer_pid": managed["process"].pid,
        "writer_start_time": 100,
    }
    try:
        await managed["owner"].start(
            {"api": 8642} if api else {},
            hermes_home=managed["home"],
            token_path=managed["token_path"],
            timeout=1,
            environment={},
            agent_config=agent,
        )
        assert managed["spawn"].await_count == 1
        assert await managed["owner"].check("ready") is None
        managed["status"].read_runtime_status.return_value["platforms"][channel][
            "state"
        ] = "retrying"
        assert await managed["owner"].check("ready") == "endpoint_unavailable"
    finally:
        await managed["owner"].stop()


@pytest.mark.parametrize(
    "channel,missing,field",
    [
        ("telegram", "TELEGRAM_BOT_TOKEN", "token_env"),
        ("slack", "SLACK_BOT_TOKEN", "bot_token_env"),
        ("slack", "SLACK_APP_TOKEN", "app_token_env"),
        ("weixin", "WEIXIN_TOKEN", "token_env"),
        ("weixin", "WEIXIN_ACCOUNT_ID", "account_id_env"),
    ],
)
async def test_channel_credentials_fail_before_creating_native_state(
    tmp_path, monkeypatch, channel, missing, field
):
    for variable in [
        "TELEGRAM_BOT_TOKEN",
        "SLACK_BOT_TOKEN",
        "SLACK_APP_TOKEN",
        "WEIXIN_TOKEN",
        "WEIXIN_ACCOUNT_ID",
    ]:
        monkeypatch.setenv(variable, "fixture-only-credential")
    monkeypatch.delenv(missing)
    runtime = HermesRuntime()
    with pytest.raises(LifecycleError) as caught:
        await runtime.start(
            {
                "base_dir": str(tmp_path),
                "config": AgentConfig.from_mapping(
                    _config({channel: {"enabled": True}})
                ),
                "runtime_context": {
                    "runtime_id": "channel-preflight",
                    "invocation_id": "start",
                    "request_id": "start",
                    "artifacts": {},
                    "environment": {
                        "environment_id": "local",
                        "provider": "local",
                        "ownership": "caller_owned",
                        "control_location": "in_env_control",
                    },
                },
            }
        )
    assert caught.value.code == f"hermes_{channel}_credential_missing"
    assert (
        caught.value.metadata["field"]
        == f"harness.settings.platforms.{channel}.{field}"
    )
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize(
    "channel,module",
    [
        ("telegram", "telegram"),
        ("slack", "slack_bolt"),
        ("slack", "slack_sdk"),
        ("weixin", "aiohttp"),
        ("weixin", "cryptography"),
    ],
)
async def test_missing_channel_dependency_fails_before_native_state(
    tmp_path, monkeypatch, channel, module
):
    for variable in [
        "TELEGRAM_BOT_TOKEN",
        "SLACK_BOT_TOKEN",
        "SLACK_APP_TOKEN",
        "WEIXIN_TOKEN",
        "WEIXIN_ACCOUNT_ID",
    ]:
        monkeypatch.setenv(variable, "fixture-only-credential")
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: None if name == module else MagicMock(),
    )
    runtime = HermesRuntime()
    with pytest.raises(LifecycleError) as caught:
        await runtime.start(
            {
                "base_dir": str(tmp_path),
                "config": AgentConfig.from_mapping(
                    _config({channel: {"enabled": True}})
                ),
                "runtime_context": {
                    "runtime_id": "channel-preflight",
                    "invocation_id": "start",
                    "request_id": "start",
                    "artifacts": {},
                    "environment": {
                        "environment_id": "local",
                        "provider": "local",
                        "ownership": "caller_owned",
                        "control_location": "in_env_control",
                    },
                },
            }
        )
    assert caught.value.code == f"hermes_{channel}_dependency_missing"
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("api", [False, True])
async def test_all_channels_share_one_gateway_and_each_must_be_connected(
    managed_gateway, api
):
    managed = managed_gateway
    names = ["discord", "telegram", "slack", "weixin"]
    statuses = managed["status"].read_runtime_status.return_value["platforms"]
    for name in names:
        statuses[name] = {
            "state": "connected",
            "writer_pid": managed["process"].pid,
            "writer_start_time": 100,
        }
    agent = AgentConfig.from_mapping(
        _config({name: {"enabled": True} for name in names})
    )
    try:
        await managed["owner"].start(
            {"api": 8642} if api else {},
            hermes_home=managed["home"],
            token_path=managed["token_path"],
            timeout=1,
            environment={},
            agent_config=agent,
        )
        assert managed["spawn"].await_count == 1
        for name in names:
            statuses[name]["state"] = "retrying"
            assert await managed["owner"].check("ready") == "endpoint_unavailable"
            statuses[name]["state"] = "connected"
            assert await managed["owner"].check("ready") is None
    finally:
        await managed["owner"].stop()
