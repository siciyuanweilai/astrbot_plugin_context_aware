"""
AstrBot 上下文场景感知增强插件 v2.1 (Context-Aware Enhancement)

为 LLM 提供结构化的群聊场景描述，增强其对对话情境的理解能力。
重点解决：主动回复时 Bot 误以为别人在问自己的问题。

核心功能:
- 触发类型检测: 被@、被回复、唤醒词、主动搭话
- 对话对象推断: 谁在和谁说话（关键功能）
- 对话流分析: 最近的对话结构
- Bot 状态追踪: 上次发言时间和内容

设计原则:
- 纯规则分析，零额外 LLM 调用
- 只做加法，不修改框架原有信息
- 与框架 LongTermMemory 协作而非冲突
- 轻量高效，不影响响应速度

Author: AstrBot
Version: 2.1.0
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

from astrbot import logger
from astrbot.api import star
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import At, Image, Plain, Reply
from astrbot.api.platform import MessageType
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.core.agent.message import TextPart

if TYPE_CHECKING:
    from astrbot.core.config import AstrBotConfig


# ============================================================================
# Constants
# ============================================================================

# 触发类型常量
TRIGGER_PRIVATE: Final = "private_chat"
TRIGGER_AT: Final = "at_bot"
TRIGGER_REPLY: Final = "reply_to_bot"
TRIGGER_WAKE: Final = "wake_word"
TRIGGER_MENTION: Final = "mention"
TRIGGER_ACTIVE: Final = "active"
TRIGGER_UNKNOWN: Final = "unknown"

# 回复特征词（用于判断是否在回复 Bot）
REPLY_STARTERS: Final = frozenset({
    "好的", "好", "嗯", "是的", "对", "谢谢", "感谢", "收到",
    "明白", "知道了", "了解", "可以", "行", "没问题",
    "ok", "OK", "Ok", "好滴", "好哒", "好嘞", "okok",
})


# ============================================================================
# Data Structures
# ============================================================================


@dataclass(slots=True)
class MessageRecord:
    """轻量级消息记录"""

    msg_id: str
    sender_id: str
    sender_name: str
    content: str
    timestamp: float  # Unix timestamp
    is_bot: bool = False
    at_bot: bool = False
    reply_to_id: str | None = None
    talking_to: str = "group"
    talking_to_name: str = "群聊"
    at_targets: list[tuple[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class SessionState:
    """会话状态 - 每个群聊/私聊一个"""

    messages: list[MessageRecord] = field(default_factory=list)
    bot_last_spoke_at: float = 0.0
    bot_last_content: str = ""


# ============================================================================
# Session Manager (LRU Cache)
# ============================================================================


class SessionManager:
    """会话管理器 - 带 LRU 淘汰机制"""

    __slots__ = ("_sessions", "_max_messages", "_max_sessions")

    def __init__(self, max_messages: int = 50, max_sessions: int = 100) -> None:
        self._sessions: OrderedDict[str, SessionState] = OrderedDict()
        self._max_messages = max(10, max_messages)
        self._max_sessions = max(10, max_sessions)

    def get(self, session_id: str) -> SessionState:
        """获取或创建会话状态"""
        if session_id in self._sessions:
            self._sessions.move_to_end(session_id)
            return self._sessions[session_id]

        while len(self._sessions) >= self._max_sessions:
            self._sessions.popitem(last=False)

        state = SessionState()
        self._sessions[session_id] = state
        return state

    def add_message(self, session_id: str, msg: MessageRecord) -> None:
        """添加消息到会话"""
        state = self.get(session_id)
        state.messages.append(msg)
        if len(state.messages) > self._max_messages:
            state.messages = state.messages[-self._max_messages:]

    def record_bot_response(self, session_id: str, content: str, ts: float) -> None:
        """记录 Bot 回复"""
        state = self.get(session_id)
        state.bot_last_spoke_at = ts
        state.bot_last_content = content[:100] if content else ""

    def has_session(self, session_id: str) -> bool:
        """检查会话是否存在"""
        return session_id in self._sessions


# ============================================================================
# Scene Analyzer
# ============================================================================


class SceneAnalyzer:
    """场景分析器 - 负责所有分析逻辑"""

    __slots__ = ("_bot_id", "_bot_names")

    def __init__(self, bot_id: str, bot_names: list[str] | None = None) -> None:
        self._bot_id = bot_id
        self._bot_names: tuple[str, ...] = tuple(
            n.lower() for n in (bot_names or []) if n
        )

    def extract_message(self, event: AstrMessageEvent) -> MessageRecord:
        """从事件提取消息记录"""
        sender_id = event.get_sender_id()
        msg = MessageRecord(
            msg_id=str(event.message_obj.message_id),
            sender_id=sender_id,
            sender_name=event.get_sender_name() or sender_id,
            content=event.message_str[:500],
            timestamp=time.time(),
            is_bot=(sender_id == self._bot_id),
        )

        for comp in event.get_messages():
            if isinstance(comp, At):
                qq_str = str(comp.qq)
                msg.at_targets.append((qq_str, comp.name or qq_str))
                if qq_str == self._bot_id:
                    msg.at_bot = True
            elif isinstance(comp, Reply):
                if comp.sender_id:
                    msg.reply_to_id = str(comp.sender_id)

        return msg

    def detect_trigger(
        self, event: AstrMessageEvent, msg: MessageRecord
    ) -> tuple[str, str]:
        """检测触发类型"""
        sender = msg.sender_name

        if event.is_private_chat():
            return TRIGGER_PRIVATE, f"私聊对话，{sender} 在直接和你交流"

        if msg.at_bot:
            return TRIGGER_AT, f"{sender} @了你，需要你回应"

        if msg.reply_to_id == self._bot_id:
            return TRIGGER_REPLY, f"{sender} 回复了你之前的消息"

        if event.is_at_or_wake_command and not msg.at_bot:
            return TRIGGER_WAKE, f"{sender} 使用唤醒词呼叫你"

        if self._bot_names:
            msg_lower = msg.content.lower()
            for name in self._bot_names:
                if name in msg_lower:
                    return TRIGGER_MENTION, f"{sender} 在消息中提到了你"

        if event.get_extra("_active_trigger") or event.get_extra("active_reply_triggered"):
            return TRIGGER_ACTIVE, "你是主动加入这个对话的，没有人在叫你"

        return TRIGGER_UNKNOWN, "触发原因未知"

    def infer_addressee(self, msg: MessageRecord, history: list[MessageRecord]) -> None:
        """推断消息的对话对象"""
        # 规则1: @ Bot
        if msg.at_bot:
            msg.talking_to, msg.talking_to_name = "bot", "你"
            return

        # 规则2: @ 其他人
        if msg.at_targets:
            target_id, target_name = msg.at_targets[0]
            if target_id != self._bot_id:
                msg.talking_to, msg.talking_to_name = target_id, target_name
                return

        # 规则3: 回复消息
        if msg.reply_to_id:
            if msg.reply_to_id == self._bot_id:
                msg.talking_to, msg.talking_to_name = "bot", "你"
            else:
                msg.talking_to = msg.reply_to_id
                for m in reversed(history):
                    if m.sender_id == msg.reply_to_id:
                        msg.talking_to_name = m.sender_name
                        break
                else:
                    msg.talking_to_name = msg.reply_to_id
            return

        # 规则4: 上下文推断
        if not history:
            return

        recent = [m for m in history[-5:] if m.sender_id != msg.sender_id]
        if not recent:
            return

        last = recent[-1]
        time_gap = msg.timestamp - last.timestamp

        # Bot 刚说完话且用户像是在回复
        if last.is_bot and time_gap < 60:
            if self._looks_like_reply(msg.content):
                msg.talking_to, msg.talking_to_name = "bot", "你"
                return

        # A-B-A 对话模式
        if last.talking_to == msg.sender_id:
            msg.talking_to, msg.talking_to_name = last.sender_id, last.sender_name
            return

        # 时间间隔很短，可能是在和上一个人对话
        if time_gap < 30 and not last.is_bot:
            msg.talking_to, msg.talking_to_name = last.sender_id, last.sender_name

    @staticmethod
    def _looks_like_reply(content: str) -> bool:
        """判断是否像回复"""
        stripped = content.strip()
        return any(stripped.startswith(s) for s in REPLY_STARTERS)


# ============================================================================
# Scene Generator - 核心：生成清晰的场景描述
# ============================================================================


class SceneGenerator:
    """场景描述生成器 - 生成清晰有力的场景描述"""

    __slots__ = ()

    @staticmethod
    def _escape(text: str) -> str:
        """XML 转义"""
        if not text:
            return ""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    def generate(
        self,
        trigger_type: str,
        trigger_desc: str,
        current: MessageRecord,
        flow: list[MessageRecord],
        bot_status: dict[str, float | str | bool],
        participants: list[str],
        *,
        show_flow: bool = True,
    ) -> str:
        """生成场景描述，重点强调对话对象"""
        esc = self._escape
        parts: list[str] = ["<conversation_scene>"]

        # ===== 1. 触发类型 =====
        parts.append(f'  <trigger type="{trigger_type}">{esc(trigger_desc)}</trigger>')

        # ===== 2. 当前消息分析（最重要的部分）=====
        is_talking_to_bot = current.talking_to == "bot"
        is_talking_to_group = current.talking_to == "group"

        if is_talking_to_bot:
            addressee_desc = "你（Bot）"
        elif is_talking_to_group:
            addressee_desc = "群里所有人（非特定对象）"
        else:
            addressee_desc = current.talking_to_name

        parts.append(
            f'  <current_message>'
            f'\n    <sender>{esc(current.sender_name)}</sender>'
            f'\n    <talking_to>{esc(addressee_desc)}</talking_to>'
            f'\n    <content>{esc(current.content[:80])}</content>'
            f'\n  </current_message>'
        )

        # ===== 3. 关键行为指导（重点！）=====
        instruction = self._generate_instruction(
            trigger_type, current, is_talking_to_bot, is_talking_to_group
        )
        if instruction:
            parts.append(f'  <instruction>{instruction}</instruction>')

        # ===== 4. 对话流（简化）=====
        if show_flow and len(flow) > 1:
            flow_lines: list[str] = []
            for m in flow[-5:]:
                to_name = "你" if m.talking_to == "bot" else (
                    "群" if m.talking_to == "group" else m.talking_to_name
                )
                sender = "[你]" if m.is_bot else m.sender_name
                preview = m.content[:20] + ("..." if len(m.content) > 20 else "")
                flow_lines.append(f'    <m>{esc(sender)} → {esc(to_name)}: {esc(preview)}</m>')
            parts.append(f'  <recent_flow>')
            parts.extend(flow_lines)
            parts.append('  </recent_flow>')

        # ===== 5. Bot 状态 =====
        if bot_status.get("active"):
            mins = bot_status.get("minutes_ago", 0)
            if isinstance(mins, (int, float)) and mins > 0:
                parts.append(f'  <your_last_message minutes_ago="{mins:.1f}"/>')

        # ===== 6. 参与者 =====
        if len(participants) > 1:
            parts.append(f'  <participants>{esc(", ".join(participants[:5]))}</participants>')

        parts.append("</conversation_scene>")
        return "\n".join(parts)

    @staticmethod
    def _generate_instruction(
        trigger: str,
        msg: MessageRecord,
        is_talking_to_bot: bool,
        is_talking_to_group: bool,
    ) -> str:
        """
        生成行为指导 - 这是解决"误以为在问自己"问题的关键
        """
        # 被明确呼叫的情况 - 正常回复
        if trigger in (TRIGGER_AT, TRIGGER_REPLY, TRIGGER_WAKE, TRIGGER_PRIVATE):
            return "用户在和你对话，请正常回应。"

        if trigger == TRIGGER_MENTION:
            return "用户提到了你，可以适当回应。"

        # 主动回复或未知触发 - 需要特别小心
        if trigger in (TRIGGER_ACTIVE, TRIGGER_UNKNOWN):
            if is_talking_to_bot:
                return "虽然是主动触发，但用户似乎在和你说话，可以回应。"

            if is_talking_to_group:
                return (
                    "【注意】这条消息是说给群里的，不是在问你。"
                    "你是主动加入对话的，请不要把这当作向你提问。"
                    "可以选择：1)发表自己的看法 2)补充相关信息 3)适当保持沉默。"
                )

            # 最关键的情况：A 在和 B 说话，Bot 主动插话
            return (
                f"【重要】{msg.sender_name} 正在和 {msg.talking_to_name} 对话，不是在问你！"
                f"你是主动加入的，不要把别人的问题当成问你的。"
                f"合适的做法：1)以旁观者身份补充 2)等待被问到再回答 3)保持沉默。"
            )

        return ""


# ============================================================================
# Main Plugin
# ============================================================================


class Main(star.Star):
    """
    上下文场景感知插件

    通过分析群聊消息结构，为 LLM 提供结构化的场景描述，
    帮助 Bot 更好地理解对话情境并做出恰当回应。

    v2.1 重点解决：主动回复时 Bot 误以为别人在问自己的问题。
    """

    def __init__(
        self,
        context: star.Context,
        config: AstrBotConfig | None = None,
    ) -> None:
        super().__init__(context)
        self._config = config

        self._enabled = self._cfg("enable", True)
        self._group_only = self._cfg("only_group_chat", True)

        self._sessions = SessionManager(
            max_messages=self._cfg("max_history", 50),
            max_sessions=self._cfg("max_groups", 100),
        )
        self._scene_generator = SceneGenerator()

        self._bot_id: str | None = None
        self._analyzer: SceneAnalyzer | None = None

        logger.info("[ContextAware] 插件 v2.1 已加载")

    def _cfg(self, key: str, default: object = None) -> object:
        """获取配置项"""
        if self._config is None:
            return default
        return self._config.get(key, default)

    def _should_process(self, event: AstrMessageEvent) -> bool:
        """判断是否应该处理此事件"""
        if not self._enabled:
            return False
        if self._group_only and event.is_private_chat():
            return False
        return True

    def _ensure_initialized(self, event: AstrMessageEvent) -> bool:
        """确保组件已初始化"""
        if self._analyzer is not None:
            return True

        self._bot_id = event.get_self_id()
        if not self._bot_id:
            logger.warning("[ContextAware] 无法获取 Bot ID，跳过处理")
            return False

        bot_names = self._cfg("bot_names", []) or []
        if not isinstance(bot_names, list):
            bot_names = []

        self._analyzer = SceneAnalyzer(self._bot_id, bot_names)
        logger.debug(f"[ContextAware] 初始化完成，Bot ID: {self._bot_id}")
        return True

    # -------------------------------------------------------------------------
    # Event Handlers
    # -------------------------------------------------------------------------

    @filter.platform_adapter_type(filter.PlatformAdapterType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        """监听所有消息，记录到历史"""
        if not self._should_process(event):
            return

        has_content = any(
            isinstance(c, (Plain, Image)) for c in event.get_messages()
        )
        if not has_content:
            return

        if not self._ensure_initialized(event):
            return

        msg = self._analyzer.extract_message(event)  # type: ignore
        state = self._sessions.get(event.unified_msg_origin)
        self._analyzer.infer_addressee(msg, state.messages)  # type: ignore

        self._sessions.add_message(event.unified_msg_origin, msg)

    @filter.on_llm_request(priority=-10)
    async def on_llm_request(
        self, event: AstrMessageEvent, req: ProviderRequest
    ) -> None:
        """在 LLM 请求前注入场景描述"""
        if not self._should_process(event):
            return

        if not self._ensure_initialized(event):
            return

        umo = event.unified_msg_origin
        if not self._sessions.has_session(umo):
            msg = self._analyzer.extract_message(event)  # type: ignore
            self._sessions.add_message(umo, msg)

        try:
            state = self._sessions.get(umo)
            if not state.messages:
                return

            current = state.messages[-1]
            trigger_type, trigger_desc = self._analyzer.detect_trigger(event, current)  # type: ignore

            window = int(self._cfg("dialogue_window", 8) or 8)
            flow = state.messages[-window:]

            now = time.time()
            bot_status: dict[str, float | str | bool] = {}
            if state.bot_last_spoke_at > 0:
                mins = (now - state.bot_last_spoke_at) / 60
                bot_status = {
                    "active": True,
                    "minutes_ago": round(mins, 1),
                    "content": state.bot_last_content,
                }

            participants = list({m.sender_name for m in flow if not m.is_bot})

            scene = self._scene_generator.generate(
                trigger_type=trigger_type,
                trigger_desc=trigger_desc,
                current=current,
                flow=flow,
                bot_status=bot_status,
                participants=participants,
                show_flow=bool(self._cfg("enable_dialogue_flow", True)),
            )

            req.extra_user_content_parts.append(TextPart(text=scene))

            # 额外日志，方便调试
            if trigger_type == TRIGGER_ACTIVE:
                logger.debug(
                    f"[ContextAware] 主动触发: {current.sender_name} → {current.talking_to_name}"
                )

        except Exception as e:
            logger.error(f"[ContextAware] 场景分析失败: {e}")

    @filter.on_llm_response()
    async def on_llm_response(
        self, event: AstrMessageEvent, resp: LLMResponse
    ) -> None:
        """记录 Bot 回复"""
        if not self._should_process(event):
            return

        if not resp.completion_text:
            return

        if not self._ensure_initialized(event):
            return

        now = time.time()
        umo = event.unified_msg_origin

        self._sessions.record_bot_response(umo, resp.completion_text, now)

        bot_msg = MessageRecord(
            msg_id=f"bot_{now}",
            sender_id=self._bot_id or "bot",
            sender_name="[你]",
            content=resp.completion_text[:200],
            timestamp=now,
            is_bot=True,
        )
        self._sessions.add_message(umo, bot_msg)

    async def terminate(self) -> None:
        """清理资源"""
        logger.info("[ContextAware] 插件已终止")
