"""逐个实测答辩用大模型：ID 在不在、首字多快、思考有没有真的关掉。

为什么要有这个脚本：模型清单是照百炼文档整理的，**没有一个经过你账号的
真实调用**。文档里第三方模型的 ID 写法前后还不一致（deepseek-v4-flash 与
deepseek-v4.1-flash 同时出现过）。与其答辩当天才发现某个选项报
model-not-found，不如现在花几分钱一次性跑完。

用法（项目根目录）：

    .\\.venv\\Scripts\\python.exe tools\\check_llm_models.py            # 只打印计划
    .\\.venv\\Scripts\\python.exe tools\\check_llm_models.py --run      # 真实调用
    .\\.venv\\Scripts\\python.exe tools\\check_llm_models.py --run --only qwen3.8-flash

每个模型只发一句很短的翻译，费用在几分钱量级。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from teams_voice_translator.defense.chat_models import (  # noqa: E402
    CHAT_MODEL_TABLE,
    chat_model,
    translation_problem,
)

PROBE_TEXT = "你好，我来简单介绍一下我的研究。"
# 实时翻译的可接受上限：首字超过这个数，说话人会明显感到卡顿。
FIRST_TOKEN_OK = 1.5
FIRST_TOKEN_WARN = 3.0


def probe(translator_cls, memory_cls, settings, model: str) -> dict:
    """发一句翻译，量首字延迟、全句耗时、思考字数。失败不抛，记进结果。"""
    translator = translator_cls(
        api_key=settings.get_api_key(),
        workspace_id=settings.workspace_id,
        model=model,
        proxy=settings.ws_proxy,
        proxy_mode=settings.proxy_mode,
    )
    started = time.monotonic()
    first = [0.0]
    reasoning = [0]

    def on_delta(_piece, _so_far):
        if not first[0]:
            first[0] = time.monotonic() - started

    payload = translator.build_payload(memory_cls(), "zh2en", PROBE_TEXT)
    try:
        text = translator._post_stream(
            payload, on_delta, None,
            on_reasoning=lambda chars: reasoning.__setitem__(0, chars),
        ).strip()
    except Exception as exc:  # noqa: BLE001 —— 探测脚本要把一切失败都记下来
        return {"model": model, "ok": False, "error": " ".join(str(exc).split())[:160]}
    total = time.monotonic() - started
    return {
        "model": model,
        "ok": bool(text),
        "first": first[0] or total,
        "total": total,
        "reasoning": reasoning[0],
        "sample": text[:48],
        "sent_thinking": payload.get("enable_thinking", "未发送"),
    }


def verdict(row: dict) -> str:
    if not row["ok"]:
        return "✗ 不可用"
    problem = translation_problem(row["model"])
    if problem:
        return "只能助答"
    if row["reasoning"]:
        # 发了 enable_thinking:false 还在思考 = 开关没生效，这是要修的 bug
        return "⚠ 思考没关掉"
    if row["first"] <= FIRST_TOKEN_OK:
        return "✓ 适合实时翻译"
    if row["first"] <= FIRST_TOKEN_WARN:
        return "~ 可用但偏慢"
    return "⚠ 太慢，只适合助答"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="实测答辩用大模型")
    parser.add_argument("--run", action="store_true", help="真实调用（会产生少量费用）")
    parser.add_argument("--only", action="append", default=[], help="只测指定模型，可重复")
    args = parser.parse_args(argv)

    models = [item.model for item in CHAT_MODEL_TABLE]
    if args.only:
        unknown = [m for m in args.only if chat_model(m) is None]
        if unknown:
            print(f"注意：{unknown} 不在能力表里，仍会按「不发思考参数」测试")
        models = list(args.only)

    if not args.run:
        print(f"将测试 {len(models)} 个模型，每个发一句：{PROBE_TEXT!r}")
        for model in models:
            info = chat_model(model)
            note = translation_problem(model) or ("旧版" if info and info.legacy else "")
            print(f"  · {model}  {note}")
        print("\n确认后加 --run 实际执行（每个模型约几分钱）。")
        return 0

    from teams_voice_translator.defense.memory import MeetingMemory
    from teams_voice_translator.defense.settings import DefenseSettings
    from teams_voice_translator.defense.translator import ContextTranslator

    settings = DefenseSettings()
    rows = []
    for model in models:
        print(f"测试 {model} …", flush=True)
        rows.append(probe(ContextTranslator, MeetingMemory, settings, model))

    print()
    print(f"{'模型':<22}{'结论':<16}{'首字':>7}{'全句':>7}{'思考字数':>9}  备注")
    print("-" * 90)
    for row in rows:
        if not row["ok"]:
            print(f"{row['model']:<22}{verdict(row):<16}{'-':>7}{'-':>7}{'-':>9}  {row.get('error', '')}")
            continue
        print(
            f"{row['model']:<22}{verdict(row):<16}{row['first']:>6.1f}s{row['total']:>6.1f}s"
            f"{row['reasoning']:>9}  enable_thinking={row['sent_thinking']}  「{row['sample']}」"
        )
    failed = [row["model"] for row in rows if not row["ok"]]
    if failed:
        print(f"\n{len(failed)} 个模型不可用：{failed}")
        print("通常是 ID 写法不对或账号未开通，把报错发给维护者，从 chat_models.py 里修正或删掉。")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
