# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native managed-channel credentials, access policy and retained-state boundaries."""

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
import json,os,sys
from pathlib import Path
from types import ModuleType
from nemo_fabric_adapters.hermes import native

def observe():
    from gateway.config import load_gateway_config,Platform
    from agent.secret_scope import get_secret
    config=load_gateway_config()
    result={}
    if "discord" in os.environ["QA_CHANNELS"]:
        from plugins.platforms.discord.adapter import DiscordAdapter
        adapter=DiscordAdapter(config.platforms[Platform.DISCORD])
        adapter._snapshot_gate_env()
        adapter._allowed_user_ids=adapter._get_allowed_users()
        adapter._allowed_role_ids=adapter._get_allowed_roles()
        result["discord"]={"token":adapter.config.token,"users":sorted(adapter._get_allowed_users()),"channels":sorted(adapter._get_allowed_channels()),"mention":adapter._discord_require_mention(),"outside":adapter._is_allowed_user("999999999",is_dm=True)}
    if "telegram" in os.environ["QA_CHANNELS"]:
        from plugins.platforms.telegram.adapter import TelegramAdapter
        from telegram import Chat,User,Message
        from datetime import datetime,timezone
        adapter=TelegramAdapter(config.platforms[Platform.TELEGRAM])
        outsider=Message(1,datetime.now(timezone.utc),Chat(-1001234567890,"supergroup"),from_user=User(999999999,False,"Outside"))
        result["telegram"]={"token":adapter.config.token,"chats":sorted(adapter._telegram_allowed_chats()),"mention":adapter._telegram_require_mention(),"guest":adapter._telegram_guest_mode(),"webhook":get_secret("TELEGRAM_WEBHOOK_URL"),"outside":adapter._is_user_authorized_from_message(outsider)}
    if "slack" in os.environ["QA_CHANNELS"]:
        from plugins.platforms.slack.adapter import SlackAdapter
        adapter=SlackAdapter(config.platforms[Platform.SLACK])
        result["slack"]={"token":adapter.config.token,"app_token":get_secret("SLACK_APP_TOKEN"),"channels":sorted(adapter._slack_allowed_channels()),"mention":adapter._slack_require_mention(),"outside":adapter._is_interactive_user_authorized("U99999999",channel_id="C12345678")}
    if "weixin" in os.environ["QA_CHANNELS"]:
        from plugins.platforms.weixin.adapter import WeixinAdapter
        adapter=WeixinAdapter(config.platforms[Platform.WEIXIN])
        result["weixin"]={"token":adapter._token,"account":adapter._account_id,"inside":adapter._is_dm_allowed("selected@im.wechat"),"outside":adapter._is_dm_allowed("outside@im.wechat"),"group":adapter._is_group_allowed("outside-group")}
    return result

def probe():
    from hermes_cli.env_loader import load_hermes_dotenv
    from hermes_cli.config import reload_env
    from agent.secret_scope import build_profile_secret_scope,set_secret_scope,reset_secret_scope
    result={"startup":observe()}
    load_hermes_dotenv()
    result["dotenv_reload"]=observe()
    reload_env()
    result["explicit_reload"]=observe()
    scope=build_profile_secret_scope(hermes_home=Path(os.environ["HERMES_HOME"]))
    token=set_secret_scope(scope)
    try: result["scope"]=observe()
    finally: reset_secret_scope(token)
    print(json.dumps(result))
cli=ModuleType("hermes_cli.main")
cli.main=probe
sys.modules["hermes_cli.main"]=cli
native.main()
"""


def _platforms():
    return {
        "discord": {
            "enabled": True,
            "token_env": "QA_DISCORD_TOKEN",
            "allowed_users": ["123456789"],
            "allowed_channels": ["234567891"],
            "require_mention": True,
        },
        "telegram": {
            "enabled": True,
            "token_env": "QA_TELEGRAM_TOKEN",
            "allowed_users": ["123456789"],
            "allowed_chats": ["-1001234567890"],
            "require_mention": True,
        },
        "slack": {
            "enabled": True,
            "bot_token_env": "QA_SLACK_BOT",
            "app_token_env": "QA_SLACK_APP",
            "allowed_users": ["U12345678"],
            "allowed_channels": ["C12345678"],
            "require_mention": True,
        },
        "weixin": {
            "enabled": True,
            "token_env": "QA_WEIXIN_TOKEN",
            "account_id_env": "QA_WEIXIN_ACCOUNT",
            "allowed_users": ["selected@im.wechat"],
        },
    }


def _agent(platforms, api=False):
    settings = {"sdk": {"enabled": False}, "platforms": platforms}
    if api:
        settings["interfaces"] = {"api": {"port": 8642}}
    return AgentConfig.from_mapping(
        {
            "harness": {"settings": settings},
            "models": {"default": {"provider": "nvidia", "model": "fixture-model"}},
        }
    )


@pytest.fixture(name="messaging_dependencies")
def messaging_dependencies_fixture():
    for module in ["telegram", "slack_bolt", "slack_sdk", "aiohttp", "cryptography"]:
        pytest.importorskip(module)


@pytest.mark.usefixtures("messaging_dependencies")
@pytest.mark.parametrize("source", ["ambient", "retained"])
@pytest.mark.parametrize("channel", ["telegram", "slack", "weixin", "all"])
def test_native_selected_channels_keep_credentials_and_access_after_reload(
    tmp_path, source, channel
):
    platforms = _platforms()
    if channel != "all":
        platforms = {channel: platforms[channel]}
    agent = _agent(platforms)
    configuration.write_hermes_config(agent, tmp_path, workspace=".")
    snapshot = tmp_path / "fabric-config.json"
    snapshot.write_text(json.dumps(agent.to_mapping()))
    selected = {
        "QA_DISCORD_TOKEN": "selected-discord-token",
        "QA_TELEGRAM_TOKEN": "123456789:selected-telegram-token",
        "QA_SLACK_BOT": "xoxb-selected-fixture-token",
        "QA_SLACK_APP": "xapp-selected-fixture-token",
        "QA_WEIXIN_TOKEN": "selected-weixin-token",
        "QA_WEIXIN_ACCOUNT": "selected-account@im.bot",
    }
    environment = {
        **os.environ,
        **selected,
        "HERMES_HOME": str(tmp_path),
        "QA_CHANNELS": ",".join(platforms),
    }
    conflicts = {
        "DISCORD_BOT_TOKEN": "retained-discord-token",
        "DISCORD_ALLOWED_USERS": "*",
        "DISCORD_ALLOWED_CHANNELS": "*",
        "DISCORD_REQUIRE_MENTION": "false",
        "DISCORD_ALLOWED_ROLES": "999999",
        "TELEGRAM_BOT_TOKEN": "123456789:retained-telegram-token",
        "TELEGRAM_ALLOWED_USERS": "999999999",
        "TELEGRAM_GROUP_ALLOWED_USERS": "999999999",
        "TELEGRAM_GROUP_ALLOWED_CHATS": "*",
        "TELEGRAM_ALLOWED_CHATS": "*",
        "TELEGRAM_GUEST_MODE": "true",
        "TELEGRAM_WEBHOOK_URL": "https://example.invalid/webhook",
        "TELEGRAM_REQUIRE_MENTION": "false",
        "TELEGRAM_ALLOW_ALL_USERS": "true",
        "SLACK_BOT_TOKEN": "xoxb-retained-fixture-token",
        "SLACK_APP_TOKEN": "xapp-retained-fixture-token",
        "SLACK_ALLOWED_USERS": "*",
        "SLACK_ALLOWED_CHANNELS": "*",
        "SLACK_REQUIRE_MENTION": "false",
        "SLACK_ALLOW_ALL_USERS": "true",
        "WEIXIN_TOKEN": "retained-weixin-token",
        "WEIXIN_ACCOUNT_ID": "outside-account@im.bot",
        "WEIXIN_ALLOWED_USERS": "outside@im.wechat",
        "WEIXIN_DM_POLICY": "open",
        "WEIXIN_ALLOW_ALL_USERS": "true",
        "WEIXIN_GROUP_POLICY": "open",
        "GATEWAY_ALLOWED_USERS": "*",
        "GATEWAY_ALLOW_ALL_USERS": "true",
    }
    if source == "ambient":
        environment.update(conflicts)
    else:
        (tmp_path / ".env").write_text(
            "".join(
                f"{key}={value}\n"
                for key, value in {
                    **conflicts,
                    **{key: "retained-selected-reference" for key in selected},
                }.items()
            )
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
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    observations = json.loads(result.stdout.splitlines()[-1])
    for phase in observations.values():
        if "discord" in platforms:
            assert phase["discord"] == {
                "token": selected["QA_DISCORD_TOKEN"],
                "users": ["123456789"],
                "channels": ["234567891"],
                "mention": True,
                "outside": False,
            }
        if "telegram" in platforms:
            assert phase["telegram"] == {
                "token": selected["QA_TELEGRAM_TOKEN"],
                "chats": ["-1001234567890"],
                "mention": True,
                "guest": False,
                "webhook": "",
                "outside": False,
            }
        if "slack" in platforms:
            assert phase["slack"] == {
                "token": selected["QA_SLACK_BOT"],
                "app_token": selected["QA_SLACK_APP"],
                "channels": ["C12345678"],
                "mention": True,
                "outside": False,
            }
        if "weixin" in platforms:
            assert phase["weixin"] == {
                "token": selected["QA_WEIXIN_TOKEN"],
                "account": selected["QA_WEIXIN_ACCOUNT"],
                "inside": True,
                "outside": False,
                "group": False,
            }
    for path in [tmp_path / "config.yaml", snapshot]:
        assert all(value not in path.read_text() for value in selected.values())


@pytest.mark.usefixtures("messaging_dependencies")
@pytest.mark.parametrize("channel", ["telegram", "slack", "weixin", "all"])
@pytest.mark.parametrize("api", [False, True])
async def test_runtime_selects_one_gateway_and_disables_other_native_channels(
    tmp_path, monkeypatch, channel, api
):
    for name, value in {
        "DISCORD_BOT_TOKEN": "selected-discord-token",
        "TELEGRAM_BOT_TOKEN": "123456789:telegram-token",
        "SLACK_BOT_TOKEN": "xoxb-slack-token",
        "SLACK_APP_TOKEN": "xapp-slack-token",
        "WEIXIN_TOKEN": "weixin-token",
        "WEIXIN_ACCOUNT_ID": "account@im.bot",
        "NVIDIA_API_KEY": "fixture-model-key",
    }.items():
        monkeypatch.setenv(name, value)
    platforms = {
        name: {"enabled": True}
        for name in (
            ["discord", "telegram", "slack", "weixin"]
            if channel == "all"
            else [channel]
        )
    }
    runtime = HermesRuntime()
    captured = {}

    async def inspect(*args, hermes_home, **kwargs):
        import yaml

        captured.update(yaml.safe_load((hermes_home / "config.yaml").read_text()))

    runtime._interfaces.start = AsyncMock(side_effect=inspect)
    try:
        await runtime.start(
            {
                "base_dir": str(tmp_path),
                "config": _agent(platforms, api),
                "runtime_context": {
                    "runtime_id": "native-messaging",
                    "invocation_id": "start",
                    "request_id": "start",
                    "artifacts": {},
                    "environment": {
                        "environment_id": "local",
                        "provider": "local",
                        "ownership": "caller_owned",
                        "control_location": "in_env_control",
                        "workspace": str(tmp_path),
                    },
                },
            }
        )
        runtime._interfaces.start.assert_awaited_once()
        assert captured["platforms"]["api_server"]["enabled"] is api
        for name in ["discord", "telegram", "slack", "weixin"]:
            assert captured["platforms"][name]["enabled"] is (name in platforms)
    finally:
        await runtime.stop()


async def test_each_selected_channel_requires_its_live_native_writer(tmp_path):
    import asyncio
    from nemo_fabric_adapters.hermes import interfaces

    script = """from gateway.status import write_runtime_status
import time
for name in ('discord','telegram','slack','weixin'):
    assert write_runtime_status(gateway_state='running', platform=name, platform_state='connected')
print('ready',flush=True)
time.sleep(30)
"""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        script,
        env={**os.environ, "HERMES_HOME": str(tmp_path)},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    owner = interfaces.InterfaceChildren()
    path = tmp_path / "gateway_state.json"
    owner._children = [
        interfaces._Child(
            "gateway",
            None,
            process,
            channel_status_path=path,
            channels=("discord", "telegram", "slack", "weixin"),
        )
    ]
    try:
        assert await asyncio.wait_for(process.stdout.readline(), 10) == b"ready\n"
        assert await owner.check("ready") is None
        record = json.loads(path.read_text())
        for name in ["discord", "telegram", "slack", "weixin"]:
            changed = json.loads(json.dumps(record))
            changed["platforms"][name]["writer_pid"] = process.pid + 100000
            path.write_text(json.dumps(changed))
            assert await owner.check("ready") == "endpoint_unavailable"
            changed["platforms"][name]["writer_pid"] = process.pid
            changed["platforms"][name]["writer_start_time"] = (
                int(record["start_time"]) - 100000
            )
            path.write_text(json.dumps(changed))
            assert await owner.check("ready") == "endpoint_unavailable"
            path.write_text(json.dumps(record))
            assert await owner.check("ready") is None
    finally:
        await owner.stop()
