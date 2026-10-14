"""一次录音 → 批量复刻多个音色的任务编排（纯逻辑，不依赖 Qt）。

为什么值得单独做这件事：做受控对比时，要比的是「同一条录音在不同模型/不同
参考时长下的表现」。一个一个手点，不但慢，还很容易在中途改错一个选项，
于是对比里混进了第二个变量，结论就废了。批量提交保证除了要比的那一项，
其余条件逐字相同。

顺带一个实打实的提速：同一个「参考时长 + 复刻通道」的若干模型，可以共用
一次样音转码和一次 OSS 上传。四个音色原本要传四次 OSS，现在通常只传一到两次。
"""
from __future__ import annotations

from dataclasses import dataclass

from ..api_payloads import (
    MINIMAX_MIN_SAMPLE_SECONDS,
    VOICE_PREPROCESS_MODELS,
    is_minimax_tts_model,
    tts_supports,
)

# 服务端硬上限；超过这个值复刻接口直接报错。
MAX_CLONE_SECONDS = 30.0


def clone_channel_for_model(model: str) -> str:
    """复刻通道由模型家族决定：qwen3-tts-vc 走 Base64 直传，其余走 OSS 中转。"""
    return "qwen3" if str(model).startswith("qwen3-tts-vc") else "legacy"


def minimum_sample_seconds(model: str) -> float:
    """各家对样音最短时长要求不同：MiniMax 10 秒，支持预处理的 3 秒，其余 5 秒。"""
    if is_minimax_tts_model(model):
        return MINIMAX_MIN_SAMPLE_SECONDS
    return 3.0 if str(model) in VOICE_PREPROCESS_MODELS else 5.0


@dataclass(frozen=True)
class CloneTask:
    """一个复刻任务 = 一个模型 × 一个参考时长。"""

    model: str
    seconds: float
    language: str = "en"
    preprocess: bool = False
    volume_normalization: bool = False

    @property
    def channel(self) -> str:
        return clone_channel_for_model(self.model)

    @property
    def min_seconds(self) -> float:
        return minimum_sample_seconds(self.model)

    @property
    def label(self) -> str:
        return f"{self.model} · {self.seconds:g}s"


@dataclass(frozen=True)
class SkippedTask:
    """被跳过的组合，以及为什么。界面必须显示出来。

    静默丢弃是最坏的做法：用户勾了 6 个组合、只出来 4 个音色，却没有任何
    解释，只会以为软件坏了。
    """

    model: str
    seconds: float
    reason: str


def build_tasks(
    models,
    durations,
    *,
    sample_seconds: float,
    language: str = "en",
    preprocess: bool = False,
    volume_normalization: bool = False,
) -> tuple[list[CloneTask], list[SkippedTask]]:
    """模型 × 时长 的笛卡尔积，按各模型的实际能力裁剪。

    时长会被夹到 [模型下限, min(样音实际长度, 30s)]。夹完之后重复的组合
    去重 —— 例如样音只有 18 秒时，「18 秒」和「全长」是同一个任务，
    不该真的提交两次、也不该产生两个几乎一样的音色让你去盲听。
    """
    if sample_seconds <= 0:
        raise ValueError("样音长度必须为正数")

    tasks: list[CloneTask] = []
    skipped: list[SkippedTask] = []
    seen: set[tuple[str, float]] = set()

    for model in models:
        model = str(model).strip()
        if not model:
            continue
        if not tts_supports(model, "clone"):
            skipped.append(SkippedTask(model, 0.0, "该模型不支持声音复刻"))
            continue
        usable = min(float(sample_seconds), MAX_CLONE_SECONDS)
        floor = minimum_sample_seconds(model)
        for raw in durations:
            seconds = round(min(float(raw), usable), 3)
            if seconds < floor:
                skipped.append(
                    SkippedTask(model, float(raw), f"不足 {floor:g} 秒，{model} 无法复刻")
                )
                continue
            key = (model, seconds)
            if key in seen:
                continue
            seen.add(key)
            tasks.append(
                CloneTask(
                    model=model,
                    seconds=seconds,
                    language=language,
                    # 两个开关都按「该任务自己的模型/通道」判定，不跟着对话框
                    # 当前绑定的模型走，否则批量里混入不支持的模型就会整批报错。
                    preprocess=bool(preprocess) and model in VOICE_PREPROCESS_MODELS,
                    volume_normalization=(
                        bool(volume_normalization) and clone_channel_for_model(model) == "legacy"
                    ),
                )
            )
    return tasks, skipped


def group_tasks(tasks) -> list[tuple[float, str, list[CloneTask]]]:
    """按 (参考时长, 复刻通道) 分组，每组只需转码一次、上传一次。

    保持首次出现的顺序，让界面上的进度和用户勾选的顺序对得上。
    """
    order: list[tuple[float, str]] = []
    buckets: dict[tuple[float, str], list[CloneTask]] = {}
    for task in tasks:
        key = (task.seconds, task.channel)
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(task)
    return [(seconds, channel, buckets[(seconds, channel)]) for seconds, channel in order]


def needs_oss(tasks) -> bool:
    """只要有一个任务走旧版通道，就必须先把 OSS 配好。"""
    return any(task.channel == "legacy" for task in tasks)


def describe_plan(tasks, skipped=()) -> str:
    """给界面用的一句话摘要：会建几个、要传几次 OSS、跳过了什么。"""
    if not tasks:
        return "当前勾选不会创建任何音色" + (f"（{len(skipped)} 个组合被跳过）" if skipped else "")
    groups = group_tasks(tasks)
    uploads = sum(1 for _seconds, channel, _group in groups if channel == "legacy")
    parts = [f"将创建 {len(tasks)} 个音色"]
    if uploads:
        parts.append(f"需要 {uploads} 次 OSS 上传")
    else:
        parts.append("无需 OSS")
    if skipped:
        parts.append(f"{len(skipped)} 个组合被跳过")
    return "，".join(parts)


# MiniMax 复刻完会用这段文字生成一条试听音频（按字计费，所以要短）。
MINIMAX_PREVIEW_TEXT = {
    "en": "Good morning. Let me briefly introduce my research.",
    "zh": "各位老师好，我来简单介绍一下我的研究。",
}


def minimax_voice_name(language: str, *, now_ms: int | None = None, salt: int = 0) -> str:
    """MiniMax 的 voice_id 要调用方自己起：字母开头、8~256 位、仅字母数字-_。

    带毫秒时间戳和序号：批量复刻同一毫秒内提交多个也不会重名 ——
    重名会直接覆盖掉前一个音色。
    """
    import time as _time

    stamp = int(_time.time() * 1000) if now_ms is None else int(now_ms)
    tag = "EN" if str(language).lower().startswith("en") else "ZH"
    return f"defense{tag}{stamp}n{int(salt)}"
