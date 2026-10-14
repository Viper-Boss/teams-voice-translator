"""答辩用大模型的能力表：翻译和助答分别能用哪些、思考模式怎么控制。

为什么需要它（2026-09 核查百炼文档后的结论）：

  **新一代模型默认开启思考。** qwen3.8-max / qwen3.8-flash / qwen3.7-plus /
  qwen3.7-flash、deepseek-v4、glm-5.2 在百炼上默认都会先「思考」再作答。
  思考内容走 `reasoning_content` 字段，正文 `content` 要等思考完才开始吐。

  我们的翻译请求原来**没有关思考**。于是只要在下拉框里选了这些新模型，
  实时翻译就会先沉默几秒到几十秒 —— 界面没字、语音不出，看起来像卡死；
  思考太久还可能撞上 60 秒的超时。对实时口译来说这是致命的。

  另外 kimi-k3、GLM-5.3 这类「仅思考」模型**关不掉思考**，根本不适合实时翻译。

所以每个模型要记三件事：思考默认开不开、用什么参数关、能不能关。
未知模型一律**什么都不发**（保持原行为），宁可慢也不要因为多发字段整条报错。
"""
from __future__ import annotations

from dataclasses import dataclass

# 思考开关的写法
TOGGLE_ENABLE_THINKING = "enable_thinking"   # 顶层 enable_thinking: bool（Qwen / DeepSeek / GLM）
TOGGLE_NONE = "none"                         # 不支持开关：要么本来就不思考，要么仅思考

# 用途
USE_TRANSLATE = "translate"
USE_QA = "qa"


@dataclass(frozen=True)
class ChatModel:
    model: str
    blurb: str
    thinks_by_default: bool
    toggle: str = TOGGLE_ENABLE_THINKING
    thinking_only: bool = False
    json_mode: bool = True
    context_k: int = 1000       # 上下文，单位千 token
    legacy: bool = False
    third_party: bool = False

    @property
    def can_translate(self) -> bool:
        """实时翻译要求：能保证不思考。仅思考模型、以及默认思考却没有已知开关的，都不行。"""
        if self.thinking_only:
            return False
        if self.thinks_by_default and self.toggle == TOGGLE_NONE:
            return False
        return True


# 顺序即各下拉框顺序：同厂商放在一起；厂商内按旗舰/均衡/高速、
# 同档按新版本优先排列。旧 ID 留在原厂商组内，旧配置仍可选。
CHAT_MODEL_TABLE: tuple[ChatModel, ...] = (
    # ---- 通义千问 当前代（百炼推荐表，2026-09）----
    ChatModel("qwen3.8-max", "最新最强：助答首选；翻译用会偏慢", True),
    ChatModel("qwen3.7-plus", "均衡：能力与成本折中", True),
    ChatModel("qwen3.8-flash", "最新高速：翻译首选（本工具自动关闭思考保证低延迟）", True),
    ChatModel("qwen3.7-flash", "上一代高速，更便宜", True),
    # ---- 通义千问旧版：仍可调用，保留旧配置 ----
    ChatModel("qwen3-max", "Qwen3 旗舰（已列为旧版）", False, legacy=True),
    ChatModel("qwen-max", "更早的旗舰", False, legacy=True),
    ChatModel("qwen-plus", "旧版均衡（此前的默认值，稳定可用）", False, legacy=True),
    ChatModel("qwen-flash", "旧版高速", False, legacy=True),
    ChatModel("qwen-turbo", "旧版最快最省", False, legacy=True),
    # ---- DeepSeek：旗舰 → 高速 ----
    ChatModel("deepseek-v4-pro", "DeepSeek V4 旗舰（第三方）", True, third_party=True),
    ChatModel("deepseek-v4-flash", "DeepSeek V4 高速（第三方）", True, third_party=True),
    # ---- 其他厂商，各自成组 ----
    ChatModel("glm-5.2", "智谱 GLM-5.2（第三方）", True, third_party=True),
    # 仅思考：答得深但慢，只适合助答
    ChatModel("kimi-k3", "Kimi K3（第三方；仅思考，只能助答，较慢）", True,
              toggle=TOGGLE_NONE, thinking_only=True, json_mode=False, third_party=True),
    ChatModel("kimi-k2.7-code", "Kimi K2.7（第三方；偏代码，默认不思考）", False,
              toggle=TOGGLE_NONE, json_mode=False, context_k=256, third_party=True),
    # MiniMax-M3 的思考开关是另一套参数（thinking: adaptive/disabled），
    # 结构未经实测，不敢乱发 —— 所以只放进助答，不进翻译。
    ChatModel("MiniMax-M3", "MiniMax M3（第三方；自适应思考，仅助答）", True,
              toggle=TOGGLE_NONE, json_mode=False, context_k=192, third_party=True),
    ChatModel("mimo-v2.5-pro", "小米 MiMo（第三方；思考默认值文档未写明，按会思考处理）",
              True, third_party=True),
)

_BY_ID = {item.model: item for item in CHAT_MODEL_TABLE}


def chat_model(model: str) -> ChatModel | None:
    return _BY_ID.get(str(model).strip())


def models_for(use: str) -> list[tuple[str, str]]:
    """给下拉框用的 (模型ID, 说明) 列表。翻译列表剔除关不掉思考的模型。"""
    rows = []
    for item in CHAT_MODEL_TABLE:
        if use == USE_TRANSLATE and not item.can_translate:
            continue
        rows.append((item.model, item.blurb))
    return rows


def thinking_fields(model: str, *, want_thinking: bool, budget: int = 0) -> dict:
    """算出请求体里要追加的思考相关字段。

    规则：
      - 表里没有的模型：什么都不发（保持改动前的行为）。
      - 想关思考、且模型默认会思考、且有已知开关 → enable_thinking: false。
      - 想开思考、且模型有开关 → enable_thinking: true，可附 thinking_budget。
      - 默认就不思考的模型想关：不发（多发字段在旧模型上有报错风险）。
    """
    info = chat_model(model)
    if info is None or info.toggle != TOGGLE_ENABLE_THINKING:
        return {}
    if not want_thinking:
        return {"enable_thinking": False} if info.thinks_by_default else {}
    fields: dict = {"enable_thinking": True}
    if budget > 0:
        fields["thinking_budget"] = int(max(1, min(32768, budget)))
    return fields


def translation_problem(model: str) -> str:
    """这个模型做实时翻译有没有硬伤。空串表示没问题。"""
    info = chat_model(model)
    if info is None:
        return ""  # 未知模型不拦，保持原行为
    if info.thinking_only:
        return f"{model} 是仅思考模型，关不掉思考，实时翻译会有明显延迟"
    if info.thinks_by_default and info.toggle == TOGGLE_NONE:
        return f"{model} 默认会思考且本工具无法关闭，实时翻译会有明显延迟"
    return ""
