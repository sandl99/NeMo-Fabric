# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discord configuration admission and Hermes-native configuration mapping."""

import os

import pytest
import yaml
from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric.errors import FabricConfigError
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapters.common.lifecycle import LifecycleError
from nemo_fabric_adapters.hermes import configuration
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
        {"slack": {"enabled": False}},
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
    tmp_path, enabled
):
    os.environ["FABRIC_DISCORD_TOKEN"] = "private-discord-token"
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
async def test_discord_activation_fails_before_creating_native_state(tmp_path, api):
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

    assert caught.value.code == "hermes_discord_activation_unsupported"
    assert not (tmp_path / "artifacts").exists()
