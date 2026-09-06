"""答辩模式入口：python defense_main.py"""
from __future__ import annotations

import faulthandler
import os
import sys
import time
import traceback
import threading
import logging
from pathlib import Path


def _enable_crash_log() -> None:
    """窗口版 exe 没有 stderr：闪退时把原生/未捕获异常落到日志文件，便于事后定位。"""
    try:
        log_dir = Path(os.getenv("APPDATA", str(Path.home()))) / "TeamsVoiceTranslator"
        log_dir.mkdir(parents=True, exist_ok=True)
        handle = open(log_dir / "defense_crash.log", "a", encoding="utf-8", buffering=1)
        from teams_voice_translator import __version__

        handle.write(
            f"\n===== 启动 {time.strftime('%Y-%m-%d %H:%M:%S')} v{__version__} exe={sys.executable} =====\n"
        )
        handle.flush()
        faulthandler.enable(handle)

        def _hook(exc_type, exc_value, exc_tb) -> None:
            handle.write(f"\n---- 未捕获异常 {time.strftime('%H:%M:%S')} ----\n")
            traceback.print_exception(exc_type, exc_value, exc_tb, file=handle)
            handle.flush()

        sys.excepthook = _hook
        def _thread_hook(args):
            handle.write(f"\n---- 后台线程异常 {args.thread.name} ----\n")
            _hook(args.exc_type, args.exc_value, args.exc_traceback)
        threading.excepthook = _thread_hook
        logger = logging.getLogger("defense.lifecycle")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(handle)
        handler.setFormatter(logging.Formatter("%(asctime)s %(threadName)s %(message)s"))
        logger.addHandler(handler)
        from PySide6.QtCore import qInstallMessageHandler
        def _qt_message(kind, context, message):
            logger.warning("Qt %s %s", kind, message)
        qInstallMessageHandler(_qt_message)
    except Exception:
        pass


if not any(flag in sys.argv for flag in ("--smoke-test", "--self-check-stress")):
    _enable_crash_log()
else:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

if __name__ == "__main__":
    if "--self-check-stress" in sys.argv:
        from teams_voice_translator.defense.selfcheck_stress import run_stress
        sys.exit(run_stress())
    from teams_voice_translator.defense.ui import run_app

    sys.exit(run_app())
