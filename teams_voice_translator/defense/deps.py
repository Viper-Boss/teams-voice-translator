"""运行时检查并在线安装可选依赖（目前只有读 PDF 用的 pypdf）。

两种运行方式分开处理：

* **打包好的 EXE**：pypdf 在打包时就被打进去了（build_defense.ps1 会先装齐依赖，
  DefenseMode.spec 收录整个 pypdf），别人拿到 EXE 直接能用，不需要任何配置。
  EXE 里没法再 pip 安装，所以万一缺了只能提示改导入 Word 版或换新版 EXE。
* **源码运行**（.venv 里的 python）：缺了就在程序里征得同意后在线安装到当前
  解释器，先用清华镜像（与 setup.ps1 一致），失败再试官方源。

安装只动当前虚拟环境，不碰系统 Python；全程在后台线程，不会卡住界面。
"""
from __future__ import annotations

import importlib
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable

PYPDF_SPEC = "pypdf>=5,<7"
PYPDF_MIN_MAJOR = 5
MIRRORS = (
    "https://pypi.tuna.tsinghua.edu.cn/simple",
    "https://pypi.org/simple",
)
INSTALL_TIMEOUT = 240


@dataclass
class DependencyStatus:
    state: str            # ok / missing / old
    version: str = ""

    @property
    def ok(self) -> bool:
        return self.state == "ok"


def _major(version: str) -> int:
    match = re.match(r"(\d+)", version or "")
    return int(match.group(1)) if match else 0


def pypdf_status(importer: Callable[[str], object] = importlib.import_module) -> DependencyStatus:
    try:
        module = importer("pypdf")
    except ImportError:
        return DependencyStatus("missing")
    version = str(getattr(module, "__version__", "") or "")
    if _major(version) < PYPDF_MIN_MAJOR:
        # 旧版对 Word 导出的 PDF 会丢空格，英文单词粘在一起，术语和检索都会变差
        return DependencyStatus("old", version)
    return DependencyStatus("ok", version)


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def can_auto_install() -> bool:
    return not is_frozen() and bool(sys.executable)


def install_command(spec: str, index_url: str, *, proxy: str = "") -> list[str]:
    command = [
        sys.executable, "-m", "pip", "install", "--upgrade", spec,
        "-i", index_url, "--disable-pip-version-check", "--no-input",
        "--only-binary=:all:",   # 与 setup.ps1 一致：只装现成的包，不在用户电脑上编译
    ]
    if proxy:
        command += ["--proxy", proxy]
    return command


def _no_window_flags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def install_package(
    spec: str = PYPDF_SPEC,
    *,
    proxy: str = "",
    on_status: Callable[[str], None] | None = None,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> tuple[bool, str]:
    """依次试各个源，装上就返回。返回 (是否成功, 最后一段输出供排错)。"""
    notify = on_status or (lambda message: None)
    if not can_auto_install():
        return False, "打包版程序不能在线安装组件"
    last_output = ""
    for index_url in MIRRORS:
        host = index_url.split("/")[2]
        notify(f"正在从 {host} 在线安装 {spec}…")
        try:
            result = runner(
                install_command(spec, index_url, proxy=proxy),
                capture_output=True, text=True, timeout=INSTALL_TIMEOUT,
                creationflags=_no_window_flags(),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            last_output = str(exc)
            continue
        last_output = ((result.stdout or "") + (result.stderr or "")).strip()[-800:]
        if result.returncode == 0:
            importlib.invalidate_caches()
            return True, last_output
    return False, last_output


def load_pypdf():
    """安装后在同一进程里重新导入。之前已加载过旧版时返回 None（需要重启程序）。"""
    importlib.invalidate_caches()
    already = sys.modules.get("pypdf")
    if already is not None:
        if _major(str(getattr(already, "__version__", ""))) >= PYPDF_MIN_MAJOR:
            return already
        return None
    try:
        return importlib.import_module("pypdf")
    except ImportError:
        return None
