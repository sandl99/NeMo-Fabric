# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Admission of managed Hermes channel configuration through Fabric planning."""

import pytest
from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric.errors import FabricConfigError


def _config(platforms, *, sdk=True):
    return FabricConfig.from_mapping(
        {
            "metadata": {"name": "channel-configuration"},
            "harness": {
                "adapter_id": "nvidia.fabric.hermes",
                "settings": {"platforms": platforms, "sdk": {"enabled": sdk}},
            },
            "models": {"default": {"provider": "nvidia", "model": "test-model"}},
        }
    )


@pytest.mark.parametrize(
    "platforms",
    [
        {
            "telegram": {
                "enabled": True,
                "token_env": "MANAGED_TELEGRAM_TOKEN",
                "allowed_users": ["123456789"],
                "allowed_chats": ["-1001234567890", "123456789"],
                "require_mention": True,
            }
        },
        {
            "slack": {
                "enabled": True,
                "bot_token_env": "MANAGED_SLACK_BOT_TOKEN",
                "app_token_env": "MANAGED_SLACK_APP_TOKEN",
                "allowed_users": ["U0123456789", "W0123456789"],
                "allowed_channels": ["C0123456789", "G0123456789"],
                "require_mention": True,
            }
        },
        {
            "weixin": {
                "enabled": True,
                "token_env": "MANAGED_WEIXIN_TOKEN",
                "account_id_env": "MANAGED_WEIXIN_ACCOUNT_ID",
                "allowed_users": ["peer@im.wechat"],
            }
        },
    ],
)
@pytest.mark.parametrize("sdk", [False, True])
def test_plan_accepts_channel_configuration_without_credentials(
    tmp_path, platforms, sdk
):
    Fabric().plan(_config(platforms, sdk=sdk), base_dir=tmp_path)


def test_plan_accepts_multiple_selected_channels_on_one_runtime(tmp_path):
    Fabric().plan(
        _config(
            {
                name: {"enabled": True}
                for name in ("discord", "telegram", "slack", "weixin")
            },
            sdk=False,
        ),
        base_dir=tmp_path,
    )


@pytest.mark.parametrize(
    "platforms",
    [
        {"telegram": {"token": "inline-credential"}},
        {"telegram": {"token_env": "bad-env-name"}},
        {"telegram": {"allowed_users": ["*"]}},
        {"telegram": {"allowed_users": ["-123"]}},
        {"telegram": {"allowed_chats": ["*"]}},
        {"telegram": {"allowed_chats": ["-0"]}},
        {"telegram": {"allowed_chats": [123]}},
        {"telegram": {"require_mention": "false"}},
        {"slack": {"bot_token": "inline-credential"}},
        {"slack": {"app_token": "inline-credential"}},
        {"slack": {"app_token_env": "bad-env-name"}},
        {"slack": {"allowed_users": ["*"]}},
        {"slack": {"allowed_channels": ["general"]}},
        {"weixin": {"token": "inline-credential"}},
        {"weixin": {"account_id": "inline-account"}},
        {"weixin": {"allowed_users": ["*"]}},
        {"weixin": {"allowed_users": [""]}},
        {"weixin": {"allowed_users": ["peer with spaces"]}},
        {"weixin": {"dm_policy": "open"}},
        {"signal": {"enabled": True}},
    ],
)
def test_plan_rejects_raw_credentials_and_unsupported_channel_settings(
    tmp_path, platforms
):
    with pytest.raises(FabricConfigError):
        Fabric().plan(_config(platforms), base_dir=tmp_path)


@pytest.mark.parametrize("channel", ["telegram", "slack", "weixin"])
@pytest.mark.parametrize("settings", [{}, {"enabled": False}])
def test_unselected_channels_do_not_create_an_execution_path(
    tmp_path, channel, settings
):
    with pytest.raises(FabricConfigError):
        Fabric().plan(_config({channel: settings}, sdk=False), base_dir=tmp_path)
