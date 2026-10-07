from __future__ import annotations

import asyncio
import logging
import uuid
from concurrent.futures import CancelledError
from pathlib import Path
from typing import Any

import acp
from acp.schema import (
    AgentCapabilities,
    AvailableCommand,
    AvailableCommandsUpdate,
    ClientCapabilities,
    CloseSessionResponse,
    CurrentModeUpdate,
    ForkSessionResponse,
    EnvVarAuthMethod,
    AuthEnvVar,
    AuthenticateResponse,
    Implementation,
    InitializeResponse,
    ListSessionsResponse,
    LoadSessionResponse,
    NewSessionResponse,
    PromptCapabilities,
    PromptResponse,
    ResumeSessionResponse,
    SessionCapabilities,
    SessionForkCapabilities,
    SessionInfo,
    SessionListCapabilities,
    SessionMode,
    SessionModeState,
    SessionResumeCapabilities,
    SetSessionConfigOptionResponse,
    SetSessionModelResponse,
    SetSessionModeResponse,
    UnstructuredCommandInput,
    Usage,
)

from agent import Agent, AgentContext, AgentContextType, UserMessage
from helpers import git, message_queue as mq, persist_chat, tokens
from helpers.localization import Localization
from initialize import initialize_agent
from usr.plugins.a0_acp import AGENT_ZERO_ACP_VERSION, PLUGIN_VERSION
from usr.plugins.a0_acp.helpers import bridge
from usr.plugins.a0_acp.helpers.content import (
    normalize_cwd,
    normalize_path_for_compare,
    prompt_blocks_to_user_message,
    stringify_message_content,
)


logger = logging.getLogger(__name__)


class AgentZeroACPAgent(acp.Agent):
    _ADVERTISED_COMMANDS = (
        {
            "name": "help",
            "description": "List Agent Zero ACP commands",
        },
        {
            "name": "context",
            "description": "Show conversation and context-window status",
        },
        {
            "name": "reset",
            "description": "Clear the current Agent Zero conversation",
        },
        {
            "name": "version",
            "description": "Show Agent Zero and ACP plugin versions",
        },
    )

    _MODE_DEFAULT = bridge.DEFAULT_MODE
    _MODE_PLAN = "plan"
    _MODE_ACT = "act"

    _MODE_INSTRUCTIONS = {
        _MODE_PLAN: (
            "ACP session mode: plan first. Prefer analysis, architecture, and tradeoffs. "
            "Do not modify files unless the user explicitly asks you to proceed."
        ),
        _MODE_ACT: (
            "ACP session mode: act. Complete the requested work end-to-end with focused "
            "implementation and validation."
        ),
    }

    def __init__(self) -> None:
        super().__init__()
        self._conn: acp.Client | None = None

    def on_connect(self, conn: acp.Client) -> None:
        self._conn = conn
        logger.info("ACP client connected")

    async def initialize(
        self,
        protocol_version: int | None = None,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        client_name = client_info.name if client_info else "unknown"
        logger.info("ACP initialize from %s (protocol v%s)", client_name, protocol_version)
        return InitializeResponse(
            protocol_version=acp.PROTOCOL_VERSION,
            agent_info=Implementation(
                name="agent-zero",
                title="Agent Zero",
                version=AGENT_ZERO_ACP_VERSION,
            ),
            agent_capabilities=AgentCapabilities(
                load_session=True,
                prompt_capabilities=PromptCapabilities(
                    embedded_context=True,
                    image=True,
                    audio=True,
                ),
                session_capabilities=SessionCapabilities(
                    fork=SessionForkCapabilities(),
                    list=SessionListCapabilities(),
                    resume=SessionResumeCapabilities(),
                ),
            ),
            auth_methods=self._auth_methods(),
        )

    def _auth_methods(self) -> list[EnvVarAuthMethod]:
        """Advertise env-var auth when an API key is configured.

        AGENT_ZERO_API_KEY gates the underlying Agent Zero instance. When it
        is set, clients must present it (bearer) to create sessions. When
        unset (local/single-user installs), no auth method is advertised and
        behavior is unchanged.
        """
        import os

        if not os.getenv("AGENT_ZERO_API_KEY"):
            return []
        return [
            EnvVarAuthMethod(
                type="env_var",
                id="agent-zero-api-key",
                name="Agent Zero API Key",
                description="Present the Agent Zero instance API key (AGENT_ZERO_API_KEY).",
                vars=[
                    AuthEnvVar(
                        name="AGENT_ZERO_API_KEY",
                        label="Agent Zero API key",
                        secret=True,
                        optional=False,
                    )
                ],
            )
        ]

    async def authenticate(self, method_id: str, **kwargs: Any) -> AuthenticateResponse | None:
        import os

        expected = os.getenv("AGENT_ZERO_API_KEY")
        if not expected:
            # No key configured: nothing to authenticate against.
            return AuthenticateResponse()
        supplied = ""
        conn = getattr(self, "_conn", None)
        headers = getattr(conn, "_headers", None) if conn else None
        if isinstance(headers, dict):
            supplied = str(headers.get("Authorization", "") or "")
            if supplied.lower().startswith("bearer "):
                supplied = supplied[7:]
        if supplied != expected:
            raise PermissionError("Agent Zero API key authentication failed")
        return AuthenticateResponse()

    async def new_session(
        self,
        cwd: str,
        additional_directories: list[str] | None = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        context = self._create_context(cwd=cwd, additional_directories=additional_directories)
        self._register(context)
        await self._send_session_start_updates(context)
        persist_chat.save_tmp_chat(context)
        return NewSessionResponse(session_id=context.id, modes=self._session_modes(context))

    async def load_session(
        self,
        cwd: str,
        session_id: str,
        additional_directories: list[str] | None = None,
        **kwargs: Any,
    ) -> LoadSessionResponse | None:
        context = self._get_context(session_id)
        if context is None:
            logger.warning("ACP load_session: missing session %s", session_id)
            return None
        self._apply_workspace(context, cwd=cwd, additional_directories=additional_directories)
        self._register(context)
        await self._replay_history(context)
        await self._send_session_start_updates(context)
        persist_chat.save_tmp_chat(context)
        return LoadSessionResponse(modes=self._session_modes(context))

    async def resume_session(
        self,
        cwd: str,
        session_id: str,
        additional_directories: list[str] | None = None,
        **kwargs: Any,
    ) -> ResumeSessionResponse:
        context = self._get_context(session_id)
        if context is None:
            context = self._create_context(
                cwd=cwd,
                additional_directories=additional_directories,
                context_id=session_id,
            )
        else:
            self._apply_workspace(context, cwd=cwd, additional_directories=additional_directories)
        self._register(context)
        await self._replay_history(context)
        await self._send_session_start_updates(context)
        persist_chat.save_tmp_chat(context)
        return ResumeSessionResponse(modes=self._session_modes(context))

    async def fork_session(
        self,
        cwd: str,
        session_id: str,
        additional_directories: list[str] | None = None,
        **kwargs: Any,
    ) -> ForkSessionResponse:
        original = self._get_context(session_id)
        if original is None:
            return ForkSessionResponse(session_id="")

        new_ids = persist_chat.load_json_chats([persist_chat.export_json_chat(original)])
        new_id = new_ids[0] if new_ids else ""
        context = self._get_context(new_id) if new_id else None
        if context is None:
            return ForkSessionResponse(session_id="")

        context.name = f"{original.name or 'ACP session'} (fork)"
        self._apply_workspace(context, cwd=cwd, additional_directories=additional_directories)
        self._register(context)
        await self._send_session_start_updates(context)
        persist_chat.save_tmp_chat(context)
        return ForkSessionResponse(session_id=context.id, modes=self._session_modes(context))

    async def list_sessions(
        self,
        cursor: str | None = None,
        cwd: str | None = None,
        **kwargs: Any,
    ) -> ListSessionsResponse:
        persist_chat.load_tmp_chats()
        contexts = [ctx for ctx in AgentContext.all() if ctx.get_data(bridge.CTX_IS_ACP)]
        if cwd:
            normalized = normalize_path_for_compare(cwd)
            contexts = [
                ctx
                for ctx in contexts
                if normalize_path_for_compare(ctx.get_data(bridge.CTX_CWD) or ctx.get_data(bridge.CTX_WORKDIR))
                == normalized
            ]

        contexts.sort(key=lambda ctx: ctx.last_message or ctx.created_at, reverse=True)
        if cursor:
            for idx, context in enumerate(contexts):
                if context.id == cursor:
                    contexts = contexts[idx + 1 :]
                    break
            else:
                contexts = []

        page = contexts[:50]
        next_cursor = contexts[50].id if len(contexts) > 50 else None
        sessions = [self._session_info(context) for context in page]
        return ListSessionsResponse(sessions=sessions, next_cursor=next_cursor)

    async def prompt(
        self,
        prompt: list[Any],
        session_id: str,
        message_id: str | None = None,
        **kwargs: Any,
    ) -> PromptResponse:
        context = self._get_context(session_id)
        if context is None:
            logger.warning("ACP prompt: missing session %s", session_id)
            return PromptResponse(stop_reason="refusal", user_message_id=message_id)

        self._register(context)
        msg_id = message_id or str(uuid.uuid4())
        parts = prompt_blocks_to_user_message(
            prompt,
            context_id=context.id,
            message_id=msg_id,
        )
        user_text = parts.text.strip()
        if not user_text and not parts.attachments:
            return PromptResponse(stop_reason="end_turn", user_message_id=msg_id)

        context.last_message = Localization.get().now()

        text_only = bool(prompt) and all(str(getattr(block, "type", "") or "") == "text" for block in prompt)
        if text_only and not parts.attachments and user_text.startswith("/"):
            response_text = await self._handle_slash_command(user_text, context)
            if response_text is not None:
                await self._send_agent_message(context.id, response_text)
                persist_chat.save_tmp_chat(context)
                return PromptResponse(stop_reason="end_turn", user_message_id=msg_id)

        if context.is_running():
            await self._send_agent_message(
                context.id,
                "A turn is already running. Wait for it to finish or send cancel from the ACP client.",
            )
            return PromptResponse(stop_reason="end_turn", user_message_id=msg_id)

        mode_instruction = self._mode_instruction(context)
        system_message = [mode_instruction] if mode_instruction else []
        mq.log_user_message(
            context,
            user_text,
            parts.attachments,
            message_id=msg_id,
            source=" (ACP)",
        )
        bridge.reset_turn(context.id, msg_id)

        task = context.communicate(
            UserMessage(
                message=user_text,
                attachments=parts.attachments,
                system_message=system_message,
                id=msg_id,
            )
        )

        stop_reason = "end_turn"
        final_text = ""
        try:
            result = await task.result()
            final_text = "" if result is None else str(result)
        except (asyncio.CancelledError, CancelledError):
            stop_reason = "cancelled"
        except Exception as exc:
            logger.exception("ACP prompt failed for session %s", session_id)
            final_text = f"Error: {exc}"
            stop_reason = "end_turn"

        session_bridge = bridge.get_bridge_for_context(context.id)
        if final_text and session_bridge and not session_bridge.response_text_sent.strip():
            await self._send_agent_message(context.id, final_text)
        elif final_text and session_bridge is None:
            await self._send_agent_message(context.id, final_text)

        persist_chat.save_tmp_chat(context)
        return PromptResponse(
            stop_reason=stop_reason,
            usage=self._usage(context, final_text),
            user_message_id=msg_id,
        )

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        context = self._get_context(session_id)
        if context:
            context.kill_process()
            bridge.reset_turn(context.id)
            logger.info("ACP cancelled session %s", session_id)

    async def close_session(self, session_id: str, **kwargs: Any) -> CloseSessionResponse | None:
        context = self._get_context(session_id)
        if context:
            context.kill_process()
            AgentContext.remove(context.id)
            persist_chat.remove_chat(context.id)
        bridge.unregister_bridge(session_id=session_id)
        return CloseSessionResponse()

    async def set_session_mode(
        self,
        mode_id: str,
        session_id: str,
        **kwargs: Any,
    ) -> SetSessionModeResponse | None:
        context = self._get_context(session_id)
        if context is None:
            return None
        normalized = str(mode_id or self._MODE_DEFAULT).strip()
        if normalized not in {self._MODE_DEFAULT, self._MODE_PLAN, self._MODE_ACT}:
            normalized = self._MODE_DEFAULT
        context.set_data(bridge.CTX_MODE, normalized)
        persist_chat.save_tmp_chat(context)
        await self._send_current_mode(context)
        return SetSessionModeResponse()

    async def set_session_model(
        self,
        model_id: str,
        session_id: str,
        **kwargs: Any,
    ) -> SetSessionModelResponse | None:
        context = self._get_context(session_id)
        if context is None:
            return None
        context.set_data(bridge.CTX_MODEL_ID, str(model_id or ""))
        persist_chat.save_tmp_chat(context)
        return SetSessionModelResponse()

    async def set_config_option(
        self,
        config_id: str,
        session_id: str,
        value: str | bool,
        **kwargs: Any,
    ) -> SetSessionConfigOptionResponse | None:
        context = self._get_context(session_id)
        if context is None:
            return None
        options = context.get_data(bridge.CTX_CONFIG_OPTIONS) or {}
        if not isinstance(options, dict):
            options = {}
        options[str(config_id)] = value
        context.set_data(bridge.CTX_CONFIG_OPTIONS, options)
        persist_chat.save_tmp_chat(context)
        return SetSessionConfigOptionResponse(config_options=[])

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method in {"ping", "health", "healthcheck"}:
            return {"ok": True}
        return {}

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        return None

    def _create_context(
        self,
        *,
        cwd: str,
        additional_directories: list[str] | None = None,
        context_id: str | None = None,
    ) -> AgentContext:
        resolved_cwd = normalize_cwd(cwd)
        context = AgentContext(
            config=initialize_agent(),
            id=context_id or None,
            name=self._title_for_cwd(resolved_cwd),
            type=AgentContextType.USER,
        )
        self._apply_workspace(context, cwd=resolved_cwd, additional_directories=additional_directories)
        return context

    def _get_context(self, session_id: str) -> AgentContext | None:
        context = AgentContext.get(session_id)
        if context is not None:
            return context
        persist_chat.load_tmp_chats()
        return AgentContext.get(session_id)

    def _apply_workspace(
        self,
        context: AgentContext,
        *,
        cwd: str,
        additional_directories: list[str] | None = None,
    ) -> None:
        resolved_cwd = normalize_cwd(cwd)
        context.set_data(bridge.CTX_IS_ACP, True)
        context.set_data(bridge.CTX_CWD, resolved_cwd)
        context.set_data(bridge.CTX_WORKDIR, resolved_cwd)
        context.set_data(
            bridge.CTX_ADDITIONAL_DIRECTORIES,
            [normalize_cwd(path) for path in (additional_directories or []) if path],
        )
        context.set_data(bridge.CTX_MODE, context.get_data(bridge.CTX_MODE) or self._MODE_DEFAULT)
        if not context.name:
            context.name = self._title_for_cwd(resolved_cwd)

    def _register(self, context: AgentContext) -> None:
        if not self._conn:
            return
        bridge.register_bridge(
            context_id=context.id,
            session_id=context.id,
            conn=self._conn,
            loop=asyncio.get_running_loop(),
        )

    def _session_modes(self, context: AgentContext) -> SessionModeState:
        current = str(context.get_data(bridge.CTX_MODE) or self._MODE_DEFAULT)
        if current not in {self._MODE_DEFAULT, self._MODE_PLAN, self._MODE_ACT}:
            current = self._MODE_DEFAULT
        return SessionModeState(
            current_mode_id=current,
            available_modes=[
                SessionMode(
                    id=self._MODE_DEFAULT,
                    name="Default",
                    description="Use the normal Agent Zero behavior.",
                ),
                SessionMode(
                    id=self._MODE_PLAN,
                    name="Plan",
                    description="Prefer planning and analysis before changing files.",
                ),
                SessionMode(
                    id=self._MODE_ACT,
                    name="Act",
                    description="Proceed end-to-end when the request is actionable.",
                ),
            ],
        )

    def _mode_instruction(self, context: AgentContext) -> str:
        return self._MODE_INSTRUCTIONS.get(str(context.get_data(bridge.CTX_MODE) or ""), "")

    async def _send_session_start_updates(self, context: AgentContext) -> None:
        await self._send_available_commands(context)
        await self._send_current_mode(context)

    async def _send_available_commands(self, context: AgentContext) -> None:
        if not self._conn:
            return
        try:
            await self._conn.session_update(
                context.id,
                AvailableCommandsUpdate(
                    session_update="available_commands_update",
                    available_commands=self._available_commands(),
                ),
            )
        except Exception:
            logger.debug("Could not advertise ACP commands", exc_info=True)

    async def _send_current_mode(self, context: AgentContext) -> None:
        if not self._conn:
            return
        try:
            update = CurrentModeUpdate(
                session_update="current_mode_update",
                current_mode_id=str(context.get_data(bridge.CTX_MODE) or self._MODE_DEFAULT),
            )
            await self._conn.session_update(context.id, update)
        except Exception:
            logger.debug("Could not send ACP current mode", exc_info=True)

    @classmethod
    def _available_commands(cls) -> list[AvailableCommand]:
        commands: list[AvailableCommand] = []
        for spec in cls._ADVERTISED_COMMANDS:
            input_hint = spec.get("input_hint")
            commands.append(
                AvailableCommand(
                    name=spec["name"],
                    description=spec["description"],
                    input=UnstructuredCommandInput(hint=input_hint) if input_hint else None,
                )
            )
        return commands

    async def _replay_history(self, context: AgentContext) -> None:
        if not self._conn:
            return
        try:
            outputs = context.agent0.history.output()
        except Exception:
            logger.debug("Could not read Agent Zero history for ACP replay", exc_info=True)
            return
        for item in outputs:
            text = stringify_message_content(item.get("content")).strip()
            if not text:
                continue
            update = (
                acp.update_agent_message_text(text)
                if item.get("ai")
                else acp.update_user_message_text(text)
            )
            try:
                await self._conn.session_update(context.id, update)
            except Exception:
                logger.debug("Could not replay ACP history", exc_info=True)
                return

    async def _send_agent_message(self, session_id: str, text: str) -> None:
        if not self._conn or not text:
            return
        try:
            await self._conn.session_update(session_id, acp.update_agent_message_text(text))
        except Exception:
            logger.debug("Could not send ACP agent message", exc_info=True)

    async def _handle_slash_command(self, text: str, context: AgentContext) -> str | None:
        command, _, args = text.partition(" ")
        command = command.lstrip("/").lower().strip()
        args = args.strip()
        if command == "help":
            lines = ["Available commands:", ""]
            for spec in self._ADVERTISED_COMMANDS:
                lines.append(f"/{spec['name']}: {spec['description']}")
            return "\n".join(lines)
        if command == "context":
            return self._context_summary(context)
        if command == "reset":
            context.reset()
            self._apply_workspace(
                context,
                cwd=context.get_data(bridge.CTX_CWD) or context.get_data(bridge.CTX_WORKDIR),
                additional_directories=context.get_data(bridge.CTX_ADDITIONAL_DIRECTORIES) or [],
            )
            return "Conversation history cleared."
        if command == "version":
            return f"Agent Zero {git.get_version()}\nACP plugin {PLUGIN_VERSION}\nACP protocol {acp.PROTOCOL_VERSION}"
        return None

    def _context_summary(self, context: AgentContext) -> str:
        outputs = context.agent0.history.output()
        user_count = sum(1 for item in outputs if not item.get("ai"))
        assistant_count = sum(1 for item in outputs if item.get("ai"))
        history_tokens = context.agent0.history.get_tokens()
        window = context.agent0.get_data(Agent.DATA_NAME_CTX_WINDOW) or {}
        window_tokens = window.get("tokens", 0) if isinstance(window, dict) else 0
        cwd = context.get_data(bridge.CTX_CWD) or context.get_data(bridge.CTX_WORKDIR) or ""
        lines = [
            f"Session: {context.id}",
            f"Workspace: {cwd}",
            f"Messages: {user_count} user, {assistant_count} assistant",
            f"History tokens: ~{history_tokens:,}",
        ]
        if window_tokens:
            lines.append(f"Last context window: ~{int(window_tokens):,} tokens")
        mode = context.get_data(bridge.CTX_MODE) or self._MODE_DEFAULT
        lines.append(f"Mode: {mode}")
        return "\n".join(lines)

    def _usage(self, context: AgentContext, final_text: str) -> Usage | None:
        window = context.agent0.get_data(Agent.DATA_NAME_CTX_WINDOW) or {}
        input_tokens = int(window.get("tokens", 0) or 0) if isinstance(window, dict) else 0
        output_tokens = tokens.approximate_tokens(final_text or "")
        total = input_tokens + output_tokens
        if total <= 0:
            return None
        return Usage(
            input_tokens=max(input_tokens, 0),
            output_tokens=max(output_tokens, 0),
            total_tokens=max(total, 0),
        )

    def _session_info(self, context: AgentContext) -> SessionInfo:
        cwd = context.get_data(bridge.CTX_CWD) or context.get_data(bridge.CTX_WORKDIR) or "."
        updated_at = context.last_message or context.created_at or Localization.get().now()
        if hasattr(updated_at, "isoformat"):
            updated_at_str = updated_at.isoformat()
        else:
            updated_at_str = str(updated_at)
        return SessionInfo(
            session_id=context.id,
            cwd=normalize_cwd(cwd),
            title=context.name or self._title_for_cwd(cwd),
            updated_at=updated_at_str,
            additional_directories=context.get_data(bridge.CTX_ADDITIONAL_DIRECTORIES) or None,
        )

    @staticmethod
    def _title_for_cwd(cwd: str) -> str:
        name = Path(str(cwd or "")).name
        return f"ACP: {name or 'workspace'}"
