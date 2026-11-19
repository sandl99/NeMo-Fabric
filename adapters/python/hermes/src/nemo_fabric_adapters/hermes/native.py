# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch a native Hermes child with Fabric's agent-construction settings.

Hermes v0.22.5's API path does not read all behavioral config from config.yaml.
Install the mapping before loading the CLI; execution and transport stay native.
The replacement is local to this adapter-owned child process.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any
from functools import wraps

from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapters.hermes import configuration
from nemo_fabric_adapters.hermes import telemetry

LOGGER = logging.getLogger(__name__)
_discord_home: Path | None = None
_discord_environment: dict[str, str] = {}


def configure_discord_environment(config: AgentConfig) -> None:
    """Keep selected native values authoritative across Hermes environment reloads."""
    global _discord_home, _discord_environment
    discord = configuration._settings(config).get("platforms", {}).get("discord", {})
    if not discord.get("enabled", False):
        return
    token_env = discord.get("token_env", "DISCORD_BOT_TOKEN")
    credential = os.environ[token_env]
    selected = {token_env: credential, "DISCORD_BOT_TOKEN": credential}
    if "allowed_users" in discord:
        selected.update(
            DISCORD_ALLOWED_USERS=",".join(discord["allowed_users"]),
            DISCORD_ALLOWED_ROLES="",
            DISCORD_ALLOW_ALL_USERS="false",
            GATEWAY_ALLOWED_USERS="",
            GATEWAY_ALLOW_ALL_USERS="false",
        )
    if "allowed_channels" in discord:
        selected["DISCORD_ALLOWED_CHANNELS"] = ",".join(discord["allowed_channels"])
    if "require_mention" in discord:
        selected["DISCORD_REQUIRE_MENTION"] = str(discord["require_mention"]).lower()
    already_configured = bool(_discord_environment)
    _discord_environment = selected
    # Capture the credential before any Hermes import can reload retained .env values.
    os.environ.update(selected)
    from hermes_constants import get_hermes_home

    _discord_home = get_hermes_home().resolve()
    if already_configured:
        return
    from agent import secret_scope
    from hermes_cli import config as hermes_config
    from hermes_cli import env_loader

    def owns(home: Path) -> bool:
        return home.resolve() == _discord_home

    def reapply(home: Path) -> None:
        if owns(home):
            os.environ.update(_discord_environment)

    load_dotenv = env_loader.load_hermes_dotenv

    @wraps(load_dotenv)
    def managed_dotenv(*args: Any, **kwargs: Any) -> Any:
        result = load_dotenv(*args, **kwargs)
        from hermes_constants import get_process_hermes_home

        reapply(Path(kwargs.get("hermes_home") or get_process_hermes_home()))
        return result

    reload_env = hermes_config.reload_env

    @wraps(reload_env)
    def managed_reload(*args: Any, **kwargs: Any) -> Any:
        result = reload_env(*args, **kwargs)
        reapply(get_hermes_home())
        return result

    build_scope = secret_scope.build_profile_secret_scope

    @wraps(build_scope)
    def managed_scope(hermes_home: Path, *args: Any, **kwargs: Any) -> dict[str, str]:
        result = build_scope(hermes_home, *args, **kwargs)
        if owns(Path(hermes_home)):
            result.update(_discord_environment)
        return result

    # Translate selected Fabric settings at native reload boundaries; Hermes still
    # owns credential lookup, authorization, and platform transport.
    env_loader.load_hermes_dotenv = managed_dotenv
    hermes_config.reload_env = managed_reload
    secret_scope.build_profile_secret_scope = managed_scope
    reapply(get_hermes_home())


def finalize_native_turn(agent: Any) -> None:
    """Close the native turn's Relay segment without deleting native conversation state."""
    if not os.environ.get("HERMES_NEMO_RELAY_PLUGINS_TOML"):
        return
    try:
        telemetry.finalize_native_relay_segment(agent.session_id)
    except Exception:
        # A telemetry failure must not replace an execution result or its error.
        LOGGER.exception("Hermes native turn Relay finalization failed")


def configured_kwargs(
    config: AgentConfig, native_kwargs: dict[str, Any], native_config: dict[str, Any]
) -> dict[str, Any]:
    """Apply Fabric-owned behavior while retaining native routing and identity."""
    enabled = (
        configuration.resolve_hermes_toolsets(config, native_config)
        if config.tools is not None and config.tools.enabled is not None
        else None
    )
    behavior = configuration.agent_behavior_kwargs(config, enabled)
    if config.tools is None or config.tools.enabled is None:
        behavior.pop("enabled_toolsets")
    if config.tools is None or not config.tools.blocked:
        behavior.pop("disabled_toolsets")
    overrides = {
        **(native_kwargs.get("request_overrides") or {}),
        **(behavior["request_overrides"] or {}),
    }
    result = {**native_kwargs, **behavior, "request_overrides": overrides or None}
    if config.instructions and config.instructions.system:
        result["ephemeral_system_prompt"] = config.instructions.system.content
    return result


def configure_browser_attachment() -> None:
    """Keep browser PTYs on this runtime's backend, including its default profile."""
    from hermes_cli import web_server_chat
    from nemo_fabric_adapters.hermes.terminal import frontend_command

    original = web_server_chat._resolve_chat_argv
    signature = inspect.signature(original)

    @wraps(original)
    def managed_chat(*args: Any, **kwargs: Any) -> Any:
        """Attach the prebuilt renderer only to this runtime's default profile."""
        bound = signature.bind(*args, **kwargs)
        profile = bound.arguments.get("profile")
        if profile not in {None, "", "current", "default"}:
            raise ValueError(
                "Managed Hermes browser chat supports only the runtime's profile"
            )
        # The pinned resolver skips its in-process backend for explicit profile
        # names, even 'default'. Select this process's home instead.
        bound.arguments["profile"] = "current"
        command = frontend_command()
        # Force the prebuilt branch in the native resolver to avoid PM/build.
        os.environ["HERMES_NODE"] = command[0]
        entry = Path(command[-1])
        if entry.parent.name == "dist":
            os.environ["HERMES_TUI_DIR"] = str(entry.parent.parent)
        argv, cwd, environment = original(*bound.args, **bound.kwargs)
        if not environment.get("HERMES_TUI_GATEWAY_URL"):
            raise ValueError(
                "Managed Hermes browser backend has no attachment endpoint"
            )
        return command, cwd, environment

    web_server_chat._resolve_chat_argv = managed_chat


def main() -> None:
    """Apply runtime-owned agent settings before entering the native CLI."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--fabric-config", type=Path, required=True)
    options, arguments = parser.parse_known_args()
    config = AgentConfig.from_mapping(json.loads(options.fabric_config.read_text()))

    if arguments[:2] == ["gateway", "run"]:
        configure_discord_environment(config)

    import run_agent
    from hermes_cli.config import load_config

    native_agent = run_agent.AIAgent
    signature = inspect.signature(native_agent)

    class ConfiguredAIAgent(native_agent):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Apply normalized behavior while retaining native agent identity."""
            bound = signature.bind(*args, **kwargs)
            super().__init__(
                **configured_kwargs(config, bound.arguments, load_config())
            )

        @wraps(native_agent.run_conversation)
        def run_conversation(self, *args: Any, **kwargs: Any) -> Any:
            """Run a native turn and finalize its Relay segment on every exit."""
            try:
                return super().run_conversation(*args, **kwargs)
            finally:
                finalize_native_turn(self)

    run_agent.AIAgent = ConfiguredAIAgent
    if arguments and arguments[0] in {"dashboard", "serve"}:
        configure_browser_attachment()
    from hermes_cli.main import main as hermes_main

    sys.argv = ["hermes", *arguments]
    hermes_main()


if __name__ == "__main__":
    main()
