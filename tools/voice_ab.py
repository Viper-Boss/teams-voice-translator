"""受控盲听对比：同一句话 × 多个 (模型, 音色)，随机化文件名后人工打分。

上一次 A/B 的结论是「差别不大，或没听清」—— 那次对比是无效的，因为没有
控制变量，也没有盲听。音色判断极易被先验污染：你知道哪个是新模型，就会
听出新模型更好。这个脚本强制做对：

  1. 同一句文本、同一采样率、同一音量、同一语气指令
  2. 输出文件名随机化成 a / b / c，映射表单独存盘，听完再看
  3. 生成一张评分表，三个维度分开打分

用法（在项目根目录）：

    .\\.venv\\Scripts\\python.exe tools\\voice_ab.py ^
        --text "In this work I propose a context-aware translation pipeline." ^
        --candidate qwen-audio-3.0-tts-plus:voice-en-001 ^
        --candidate cosyvoice-v3.5-plus:voice-en-002 ^
        --candidate qwen3-tts-vc-realtime-2026-01-15:voice-zh-003 ^
        --out ab-run-1

默认只打印计划。只有额外传入 --execute 才会发起真实云端请求并计费。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from teams_voice_translator.defense.speech_policy import (  # noqa: E402
    SPEECH_SAMPLE_RATE,
    speech_settings,
)

# aliyun / settings / tts_session 会拉进 websocket、凭据管理器等重依赖。
# --dry-run 只是打印计划，不该因为环境没装全就跑不起来，所以延迟到真正合成时再导入。

# 对比句默认取一句有专业术语、长度适中（25~35 词）的学术口语。
# 太短听不出韵律，太长注意力会飘。
DEFAULT_TEXT = (
    "In this work I propose a context-aware translation pipeline, and the "
    "ablation study shows it improves terminology consistency by about "
    "twelve percent over the sentence-level baseline."
)

SCORESHEET_HEADER = """# 盲听评分表

**听之前不要打开 mapping.json。**

听法：
- 同一副耳机 / 同一个音量，全程不要调
- 每个文件至少听两遍，可以来回切
- 找一个不知道你在测什么的人一起打分

| 文件 | 音色相似度(1-5) | 自然度(1-5) | 可信度(1-5) | 备注 |
|------|----------------|------------|------------|------|
"""

SCORESHEET_FOOTER = """
三个维度分别是：
- **音色相似度**：像不像你本人
- **自然度**：像不像真人在说话（和像不像你无关）
- **可信度**：如果这是一段答辩录音，你会怀疑它是合成的吗（5 = 完全不会）

打完分再打开 mapping.json 对照。
"""


def parse_candidate(raw: str) -> tuple[str, str]:
    """解析 ``model:voice_id``；模型名不含冒号，音色 ID 即使含冒号也能保留。"""
    if ":" not in raw:
        raise argparse.ArgumentTypeError(
            f"候选格式应为 模型:音色ID，收到 {raw!r}"
        )
    model, voice = raw.split(":", 1)
    model, voice = model.strip(), voice.strip()
    if not model or not voice:
        raise argparse.ArgumentTypeError(f"候选的模型或音色为空：{raw!r}")
    return model, voice


def synthesize(settings, model: str, voice: str, text: str,
               timeout: float = 90.0) -> bytes:
    """合成一句，返回裸 PCM16。失败直接抛，不要静默产出半截音频。"""
    from teams_voice_translator.aliyun import BailianClient
    from teams_voice_translator.defense.tts_session import create_tts_session

    client = BailianClient(
        settings.get_api_key(),
        settings.workspace_id,
        proxy=settings.ws_proxy,
        proxy_mode=settings.proxy_mode,
    )
    payload = speech_settings(settings, model=model, voice=voice, language="en")
    captured = bytearray()
    errors: list[str] = []
    done = threading.Event()

    session = create_tts_session(
        client,
        payload,
        on_audio=captured.extend,
        on_error=errors.append,
        on_utterance_done=lambda *_: done.set(),
        on_utterance_failed=lambda *_: done.set(),
        connect_timeout=15,
        utterance_timeout=timeout,
        max_attempts=1,
    )
    session.start()
    try:
        if not session.speak(text):
            raise RuntimeError("合成任务未能提交")
        if not done.wait(timeout):
            raise RuntimeError(f"{model} 合成超时")
    finally:
        session.close()

    if errors:
        raise RuntimeError(f"{model} 合成失败：{errors[0]}")
    if not captured:
        raise RuntimeError(f"{model} 未返回音频")
    return bytes(captured)


def write_wav(path: Path, pcm: bytes, sample_rate: int = SPEECH_SAMPLE_RATE) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="受控盲听 A/B 对比")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="对比用的英文句子")
    parser.add_argument("--candidate", action="append", type=parse_candidate,
                        required=True, metavar="模型:音色ID",
                        help="可重复，每个候选一次")
    parser.add_argument("--out", default="voice-ab", help="输出目录")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--execute",
        action="store_true",
        help="明确允许发起真实合成（会产生云端费用）",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印计划；为兼容旧命令保留，当前也是默认行为",
    )
    args = parser.parse_args(argv)

    candidates = list(args.candidate)
    if len(candidates) < 2:
        parser.error("至少需要两个候选才能做对比")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 先随机化标签再合成：合成顺序本身也不该泄露身份。
    labels = [chr(ord("a") + index) for index in range(len(candidates))]
    random.shuffle(labels)

    if not args.execute:
        print(f"将合成 {len(candidates)} 个候选，文本：{args.text!r}")
        for label, (model, voice) in zip(labels, candidates):
            print(f"  {label}.wav  ←  {model}  /  {voice}")
        print("\n确认无误后加上 --execute 实际执行（会产生云端费用）。")
        return 0

    from teams_voice_translator.defense.settings import DefenseSettings

    settings = DefenseSettings()
    mapping: dict[str, dict[str, str]] = {}
    failures: list[str] = []

    for index, (label, (model, voice)) in enumerate(zip(labels, candidates), start=1):
        # 真正执行时不要在终端泄露 label → 模型映射，否则还没听就失盲。
        print(f"正在合成候选 {index}/{len(candidates)}…", flush=True)
        try:
            pcm = synthesize(settings, model, voice, args.text)
        except Exception as exc:
            # 一个候选失败不影响其余对比，但要在结果里留痕，不能当作没发生。
            print(f"  候选 {index} 失败，详情已写入 mapping.json", file=sys.stderr)
            failures.append(f"{label}: {model} / {voice} — {exc}")
            continue
        write_wav(out_dir / f"{label}.wav", pcm)
        mapping[label] = {"model": model, "voice": voice}

    (out_dir / "mapping.json").write_text(
        json.dumps({"text": args.text, "candidates": mapping, "failures": failures},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows = "".join(f"| {label}.wav |  |  |  |  |\n" for label in sorted(mapping))
    (out_dir / "评分表.md").write_text(
        SCORESHEET_HEADER + rows + SCORESHEET_FOOTER, encoding="utf-8"
    )

    print(f"\n完成：{len(mapping)} 个候选写入 {out_dir}")
    if failures:
        print(f"其中 {len(failures)} 个失败，详见 mapping.json")
    if len(mapping) < 2:
        print("成功候选不足两个，本轮不能进行有效盲听对比。", file=sys.stderr)
        return 1
    print("先打开 评分表.md 盲听打分，打完再看 mapping.json。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
