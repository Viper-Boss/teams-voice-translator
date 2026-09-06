from __future__ import annotations

import ctypes
import hashlib
import os
import re
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import psutil
import requests

from .voicestudio import VoiceStudioClient


GITHUB_LATEST_RELEASE = "https://api.github.com/repos/debpalash/VoiceStudio/releases/latest"
GITHUB_RELEASES_PAGE = "https://github.com/debpalash/VoiceStudio/releases/latest"
_PRODUCT_CODE_RE = re.compile(r"\{[0-9A-Fa-f-]{36}\}")
# Generic Python processes are matched only by backend-specific markers.  The
# broad word "voicestudio" would also match this manager module and test names.
_PROCESS_MARKERS = ("omnivoice", "com.debpalash.omnivoice-studio")
_ROOT_PROCESS_NAMES = {
    "voicestudio.exe",
    "omnivoice.exe",
    "omnivoice-studio.exe",
    "omnivoice studio.exe",
}
_INSTALL_EXECUTABLE_NAMES = (
    "VoiceStudio.exe",
    "omnivoice-studio.exe",
    "OmniVoice.exe",
    "OmniVoice Studio.exe",
)
_CHILD_PROCESS_NAMES = {
    "python.exe",
    "pythonw.exe",
    "uv.exe",
    "node.exe",
    "bun.exe",
    "ffmpeg.exe",
    "ffprobe.exe",
    "omnivoice-sidecar.exe",
}


class VoiceStudioManagerError(RuntimeError):
    pass


class VoiceStudioTaskCancelled(VoiceStudioManagerError):
    """Raised when the user cancels a cancellable management operation."""

    pass


@dataclass(frozen=True)
class VoiceStudioInstallation:
    installed: bool = False
    version: str = ""
    install_dir: str = ""
    executable: str = ""
    uninstall_string: str = ""
    quiet_uninstall_string: str = ""
    display_name: str = ""


@dataclass(frozen=True)
class VoiceStudioRelease:
    version: str
    tag: str
    asset_name: str
    download_url: str
    size: int = 0
    page_url: str = GITHUB_RELEASES_PAGE
    expected_sha256: str = ""


@dataclass(frozen=True)
class VoiceStudioProcess:
    pid: int
    ppid: int
    name: str
    role: str
    memory_mb: float
    uptime_seconds: int
    command_line: str
    managed: bool = False


@dataclass(frozen=True)
class VoiceStudioRuntime:
    processes: tuple[VoiceStudioProcess, ...]
    api_ok: bool
    api_version: str = ""
    api_device: str = ""
    api_message: str = ""
    installation: VoiceStudioInstallation = VoiceStudioInstallation()
    latest_release: VoiceStudioRelease | None = None


ProgressCallback = Callable[[str, int], None]
StopCallback = Callable[[str], None]


def _clean_registry_command(value: Any) -> str:
    return str(value or "").strip().strip("\x00")


def _clean_display_icon(value: str) -> str:
    raw = _clean_registry_command(value)
    if raw.startswith('"') and '"' in raw[1:]:
        return raw.split('"', 2)[1]
    return raw.rsplit(",", 1)[0].strip().strip('"')


def _normalize_version(value: str) -> str:
    return str(value or "").strip().lstrip("vV")


def _format_role(name: str, command_line: str) -> str:
    haystack = f"{name} {command_line}".lower()
    if name.lower() in _ROOT_PROCESS_NAMES:
        return "桌面程序"
    if "uvicorn" in haystack or "backend.main" in haystack:
        return "本地 API"
    if "sidecar" in haystack or "worker" in haystack:
        return "模型工作进程"
    if "ffmpeg" in haystack:
        return "音频处理"
    return "后台子进程"


def parse_release_payload(payload: Any) -> VoiceStudioRelease:
    if not isinstance(payload, dict):
        raise VoiceStudioManagerError("GitHub 最新版本信息格式无效。")
    assets = payload.get("assets") if isinstance(payload.get("assets"), list) else []
    candidates: list[dict[str, Any]] = []
    for item in assets:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        lowered = name.lower()
        if lowered.endswith(".msi") and ("x64" in lowered or "amd64" in lowered):
            candidates.append(item)
    if not candidates:
        raise VoiceStudioManagerError("官方最新版本中没有找到 Windows x64 MSI 安装包。")
    asset = candidates[0]
    tag = str(payload.get("tag_name") or payload.get("name") or "").strip()
    version = _normalize_version(tag)
    url = str(asset.get("browser_download_url") or "").strip()
    if not url:
        raise VoiceStudioManagerError("官方 MSI 下载地址为空。")
    expected_sha256 = ""
    release_body = str(payload.get("body") or "")
    asset_name = str(asset.get("name") or "VoiceStudio_x64.msi")
    for checksum, hashed_name in re.findall(
        r"(?im)^\s*([0-9a-f]{64})\s+\*?([^\r\n]+\.msi)\s*$",
        release_body,
    ):
        if Path(hashed_name.strip()).name.lower() == asset_name.lower():
            expected_sha256 = checksum.lower()
            break
    return VoiceStudioRelease(
        version=version,
        tag=tag or version,
        asset_name=asset_name,
        download_url=url,
        size=int(asset.get("size") or 0),
        page_url=str(payload.get("html_url") or GITHUB_RELEASES_PAGE),
        expected_sha256=expected_sha256,
    )


class VoiceStudioManager:
    """Own the Windows VoiceStudio install and process lifecycle used by this app."""

    def __init__(self) -> None:
        self.managed_pids: set[int] = set()
        self.session_managed = False
        self._latest_cache: VoiceStudioRelease | None = None
        self._latest_checked_at = 0.0

    @staticmethod
    def _raise_if_cancelled(cancel_event: threading.Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise VoiceStudioTaskCancelled("VoiceStudio 操作已取消。")

    @staticmethod
    def _find_install_executable(directory: str | Path) -> Path | None:
        root = Path(directory).expanduser()
        for name in _INSTALL_EXECUTABLE_NAMES:
            candidate = root / name
            if candidate.is_file():
                return candidate
        return None

    @staticmethod
    def _registry_installations() -> list[VoiceStudioInstallation]:
        if os.name != "nt":
            return []
        try:
            import winreg
        except ImportError:
            return []

        roots = (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE)
        views = (0, getattr(winreg, "KEY_WOW64_64KEY", 0), getattr(winreg, "KEY_WOW64_32KEY", 0))
        base = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"
        found: list[VoiceStudioInstallation] = []
        seen: set[tuple[str, str]] = set()
        for root in roots:
            for view in views:
                try:
                    key = winreg.OpenKey(root, base, 0, winreg.KEY_READ | view)
                except OSError:
                    continue
                with key:
                    for index in range(winreg.QueryInfoKey(key)[0]):
                        try:
                            child_name = winreg.EnumKey(key, index)
                            child = winreg.OpenKey(key, child_name)
                        except OSError:
                            continue
                        with child:
                            def read(name: str) -> str:
                                try:
                                    return _clean_registry_command(winreg.QueryValueEx(child, name)[0])
                                except OSError:
                                    return ""

                            display_name = read("DisplayName")
                            if not any(marker in display_name.lower() for marker in ("voicestudio", "omnivoice")):
                                continue
                            install_dir = read("InstallLocation")
                            display_icon = _clean_display_icon(read("DisplayIcon"))
                            executable = display_icon if display_icon.lower().endswith(".exe") else ""
                            if not executable and install_dir:
                                candidate = VoiceStudioManager._find_install_executable(
                                    install_dir
                                )
                                if candidate is not None:
                                    executable = str(candidate)
                            signature = (display_name, install_dir or executable)
                            if signature in seen:
                                continue
                            seen.add(signature)
                            found.append(
                                VoiceStudioInstallation(
                                    installed=True,
                                    version=read("DisplayVersion"),
                                    install_dir=install_dir or (str(Path(executable).parent) if executable else ""),
                                    executable=executable,
                                    uninstall_string=read("UninstallString"),
                                    quiet_uninstall_string=read("QuietUninstallString"),
                                    display_name=display_name,
                                )
                            )
        return found

    def detect_installation(
        self,
        configured_executable: str = "",
        configured_install_dir: str = "",
    ) -> VoiceStudioInstallation:
        configured = Path(configured_executable).expanduser() if configured_executable else None
        registry_items = self._registry_installations()
        for item in registry_items:
            executable = Path(item.executable) if item.executable else None
            if executable is None or not executable.is_file():
                executable = (
                    self._find_install_executable(item.install_dir)
                    if item.install_dir
                    else None
                )
            if executable is not None:
                return VoiceStudioInstallation(**{**item.__dict__, "executable": str(executable)})

        candidates: list[Path | None] = [configured]
        search_directories = [
            configured_install_dir,
            Path(os.getenv("ProgramFiles", r"C:\Program Files")) / "VoiceStudio",
            Path(os.getenv("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
            / "Programs"
            / "VoiceStudio",
        ]
        candidates.extend(
            self._find_install_executable(directory)
            for directory in search_directories
            if directory
        )
        executable = next((item for item in candidates if item is not None and item.is_file()), None)
        if executable is None:
            # The MSI registration itself is authoritative for the installed
            # version even if a future release renames its desktop executable.
            if registry_items:
                return registry_items[0]
            return VoiceStudioInstallation()
        return VoiceStudioInstallation(
            installed=True,
            install_dir=str(executable.parent),
            executable=str(executable),
            display_name="VoiceStudio",
        )

    @staticmethod
    def _command_line(process: psutil.Process) -> str:
        try:
            return " ".join(process.cmdline())
        except (psutil.Error, OSError):
            return ""

    def _discover_process_objects(self) -> list[psutil.Process]:
        roots: list[psutil.Process] = []
        candidates: dict[int, psutil.Process] = {}
        for process in psutil.process_iter(["pid", "name"]):
            try:
                name = str(process.info.get("name") or "")
                command_line = self._command_line(process)
                haystack = f"{name} {command_line}".lower()
                if name.lower() in _ROOT_PROCESS_NAMES or (
                    name.lower() in _CHILD_PROCESS_NAMES
                    and any(marker in haystack for marker in _PROCESS_MARKERS)
                ):
                    roots.append(process)
                    candidates[process.pid] = process
            except (psutil.Error, OSError):
                continue
        for root in list(roots):
            try:
                for child in root.children(recursive=True):
                    candidates[child.pid] = child
            except psutil.Error:
                continue
        return list(candidates.values())

    def list_processes(self, *, managed_only: bool = False) -> list[VoiceStudioProcess]:
        processes = self._discover_process_objects()
        if managed_only:
            self._expand_managed_descendants(processes)
            processes = [item for item in processes if item.pid in self.managed_pids]
        result: list[VoiceStudioProcess] = []
        now = time.time()
        for process in processes:
            try:
                with process.oneshot():
                    name = process.name()
                    command_line = self._command_line(process)
                    result.append(
                        VoiceStudioProcess(
                            pid=process.pid,
                            ppid=process.ppid(),
                            name=name,
                            role=_format_role(name, command_line),
                            memory_mb=round(process.memory_info().rss / 1024 / 1024, 1),
                            uptime_seconds=max(0, int(now - process.create_time())),
                            command_line=command_line,
                            managed=process.pid in self.managed_pids,
                        )
                    )
            except (psutil.Error, OSError):
                continue
        return sorted(result, key=lambda item: (item.role != "桌面程序", item.pid))

    def _expand_managed_descendants(self, processes: Iterable[psutil.Process] | None = None) -> None:
        pool = list(processes) if processes is not None else self._discover_process_objects()
        by_pid = {item.pid: item for item in pool}
        changed = True
        while changed:
            changed = False
            for process in pool:
                try:
                    if process.pid not in self.managed_pids and process.ppid() in self.managed_pids:
                        self.managed_pids.add(process.pid)
                        changed = True
                except psutil.Error:
                    continue
        self.managed_pids.intersection_update({pid for pid in by_pid if psutil.pid_exists(pid)})

    def adopt_running_processes(self) -> None:
        for process in self._discover_process_objects():
            self.managed_pids.add(process.pid)
        if self.managed_pids:
            self.session_managed = True

    def has_managed_processes(self) -> bool:
        return self.session_managed and bool(self.list_processes())

    def latest_release(self, *, force: bool = False) -> VoiceStudioRelease:
        if not force and self._latest_cache is not None and time.time() - self._latest_checked_at < 1800:
            return self._latest_cache
        session = requests.Session()
        session.trust_env = True
        try:
            response = session.get(
                GITHUB_LATEST_RELEASE,
                headers={"Accept": "application/vnd.github+json", "User-Agent": "TeamsVoiceTranslator"},
                timeout=20,
            )
            response.raise_for_status()
            release = parse_release_payload(response.json())
        except (requests.RequestException, ValueError) as exc:
            raise VoiceStudioManagerError(f"读取 VoiceStudio 最新版本失败：{exc}") from exc
        self._latest_cache = release
        self._latest_checked_at = time.time()
        return release

    def runtime_snapshot(
        self,
        base_url: str,
        configured_executable: str = "",
        configured_install_dir: str = "",
        *,
        include_latest: bool = False,
    ) -> VoiceStudioRuntime:
        processes = tuple(self.list_processes())
        api_ok = False
        api_version = ""
        api_device = ""
        api_message = ""
        try:
            health = VoiceStudioClient(base_url, timeout=3).health()
            api_ok = str(health.get("status") or "ok").lower() in {"ok", "ready", "healthy"}
            api_version = str(health.get("version") or "")
            api_device = str(health.get("device") or health.get("compute") or "")
            api_message = str(health.get("message") or "")
        except Exception as exc:
            api_message = str(exc)
        release = None
        if include_latest:
            try:
                release = self.latest_release()
            except VoiceStudioManagerError:
                release = self._latest_cache
        return VoiceStudioRuntime(
            processes=processes,
            api_ok=api_ok,
            api_version=api_version,
            api_device=api_device,
            api_message=api_message,
            installation=self.detect_installation(configured_executable, configured_install_dir),
            latest_release=release,
        )

    def download_latest_installer(
        self,
        progress: ProgressCallback,
        cancel_event: threading.Event | None = None,
    ) -> tuple[Path, VoiceStudioRelease]:
        self._raise_if_cancelled(cancel_event)
        release = self.latest_release(force=True)
        self._raise_if_cancelled(cancel_event)
        target_dir = Path(tempfile.gettempdir()) / "TeamsVoiceTranslator" / "VoiceStudio"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / release.asset_name
        partial = target.with_suffix(target.suffix + ".part")
        session = requests.Session()
        session.trust_env = True
        try:
            with session.get(
                release.download_url,
                headers={"User-Agent": "TeamsVoiceTranslator"},
                stream=True,
                timeout=(15, 180),
            ) as response:
                response.raise_for_status()
                total = int(response.headers.get("Content-Length") or release.size or 0)
                downloaded = 0
                with partial.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 512):
                        self._raise_if_cancelled(cancel_event)
                        if not chunk:
                            continue
                        handle.write(chunk)
                        downloaded += len(chunk)
                        percent = int(downloaded * 100 / total) if total else -1
                        size_text = f"{downloaded / 1024 / 1024:.1f} MB"
                        if total:
                            size_text += f" / {total / 1024 / 1024:.1f} MB"
                        progress(f"正在下载 {release.asset_name} · {size_text}", percent)
            self._raise_if_cancelled(cancel_event)
            partial.replace(target)
        except VoiceStudioTaskCancelled:
            try:
                partial.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        except (requests.RequestException, OSError) as exc:
            try:
                partial.unlink(missing_ok=True)
            except OSError:
                pass
            raise VoiceStudioManagerError(f"下载 VoiceStudio 安装包失败：{exc}") from exc
        progress(f"下载完成 · {release.asset_name}", 100)
        if release.expected_sha256:
            actual = self.file_sha256(target, progress, cancel_event)
            if actual.lower() != release.expected_sha256.lower():
                raise VoiceStudioManagerError(
                    "官方下载文件的 SHA-256 与 GitHub 发布页不一致，已停止自动安装。"
                )
            progress("GitHub SHA-256 校验通过。", 100)
        return target, release

    @staticmethod
    def file_sha256(
        path: Path,
        progress: ProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
    ) -> str:
        total = max(0, path.stat().st_size)
        completed = 0
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                VoiceStudioManager._raise_if_cancelled(cancel_event)
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                completed += len(chunk)
                if progress is not None:
                    percent = int(completed * 100 / total) if total else -1
                    progress(f"正在计算 SHA-256 · {completed / 1024 / 1024:.1f} MB", percent)
        return digest.hexdigest().lower()

    def verify_manual_installer(
        self,
        installer: Path,
        progress: ProgressCallback,
        cancel_event: threading.Event | None = None,
    ) -> tuple[VoiceStudioRelease, str, bool | None]:
        if not installer.is_file() or installer.suffix.lower() != ".msi":
            raise VoiceStudioManagerError("请选择有效的 VoiceStudio Windows MSI 安装包。")
        progress("正在从 GitHub 获取最新版本和官方 SHA-256…", -1)
        self._raise_if_cancelled(cancel_event)
        release = self.latest_release(force=True)
        actual = self.file_sha256(installer, progress, cancel_event)
        matched = (
            actual.lower() == release.expected_sha256.lower()
            if release.expected_sha256
            else None
        )
        return release, actual, matched

    @staticmethod
    def _run_installer(
        arguments: list[str],
        progress: ProgressCallback,
        label: str,
        cancel_event: threading.Event | None = None,
    ) -> None:
        progress(label, -1)
        VoiceStudioManager._raise_if_cancelled(cancel_event)
        process = subprocess.Popen(arguments)
        while process.poll() is None:
            if cancel_event is not None and cancel_event.wait(0.15):
                progress("正在取消 Windows Installer 并回滚更改…", -1)
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
                raise VoiceStudioTaskCancelled("VoiceStudio 安装操作已取消。")
            if cancel_event is None:
                time.sleep(0.15)
        if process.returncode not in {0, 1641, 3010}:
            raise VoiceStudioManagerError(f"Windows Installer 返回错误代码 {process.returncode}。")

    def install(
        self,
        install_dir: str,
        progress: ProgressCallback,
        cancel_event: threading.Event | None = None,
    ) -> VoiceStudioInstallation:
        installer, release = self.download_latest_installer(progress, cancel_event)
        return self.install_package(
            installer,
            install_dir,
            progress,
            release.tag,
            cancel_event,
        )

    def install_package(
        self,
        installer: Path,
        install_dir: str,
        progress: ProgressCallback,
        version_label: str = "",
        cancel_event: threading.Event | None = None,
    ) -> VoiceStudioInstallation:
        if not installer.is_file() or installer.suffix.lower() != ".msi":
            raise VoiceStudioManagerError("安装包不存在或不是 MSI 文件。")
        arguments = ["msiexec.exe", "/i", str(installer)]
        if install_dir.strip():
            arguments.append(f"INSTALLDIR={str(Path(install_dir).expanduser())}")
        arguments.extend(["/passive", "/norestart"])
        suffix = f" {version_label}" if version_label else ""
        self._run_installer(
            arguments,
            progress,
            f"正在安装 VoiceStudio{suffix}…",
            cancel_event,
        )
        progress("安装完成，正在读取程序信息…", 100)
        return self.detect_installation("", install_dir)

    def upgrade(
        self,
        install_dir: str,
        progress: ProgressCallback,
        stop_callback: StopCallback | None = None,
        cancel_event: threading.Event | None = None,
    ) -> VoiceStudioInstallation:
        self.stop_all(progress=stop_callback or (lambda _message: None), managed_only=False)
        self._raise_if_cancelled(cancel_event)
        installer, release = self.download_latest_installer(progress, cancel_event)
        arguments = ["msiexec.exe", "/i", str(installer)]
        if install_dir.strip():
            arguments.append(f"INSTALLDIR={str(Path(install_dir).expanduser())}")
        arguments.extend(["/passive", "/norestart"])
        self._run_installer(
            arguments,
            progress,
            f"正在升级到 VoiceStudio {release.tag}…",
            cancel_event,
        )
        progress("升级完成，正在刷新版本信息…", 100)
        return self.detect_installation("", install_dir)

    def start(self, executable: str = "", install_dir: str = "") -> int:
        installation = self.detect_installation(executable, install_dir)
        path = Path(installation.executable) if installation.executable else None
        if path is None or not path.is_file():
            raise VoiceStudioManagerError("没有找到 VoiceStudio.exe，请先安装或选择正确的程序路径。")
        try:
            process = subprocess.Popen([str(path)], cwd=str(path.parent))
        except OSError as exc:
            raise VoiceStudioManagerError(f"启动 VoiceStudio 失败：{exc}") from exc
        self.managed_pids.add(process.pid)
        self.session_managed = True
        return process.pid

    @staticmethod
    def _post_close_to_windows(pids: set[int]) -> None:
        if os.name != "nt" or not pids:
            return
        user32 = ctypes.windll.user32
        enum_proc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

        def callback(hwnd: int, _lparam: int) -> bool:
            pid = ctypes.c_ulong()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if int(pid.value) in pids and user32.IsWindowVisible(hwnd):
                user32.PostMessageW(hwnd, 0x0010, 0, 0)  # WM_CLOSE
            return True

        user32.EnumWindows(enum_proc(callback), 0)

    def stop_all(self, progress: StopCallback, *, managed_only: bool = False) -> list[int]:
        # Once this app started or explicitly connected to a local VoiceStudio
        # session, include its discoverable orphan workers as well.  This keeps
        # a crashed desktop shell from leaving the model backend behind.
        process_infos = self.list_processes(
            managed_only=managed_only and not self.session_managed
        )
        if not process_infos:
            return []
        processes: list[psutil.Process] = []
        for info in process_infos:
            try:
                processes.append(psutil.Process(info.pid))
            except psutil.Error:
                continue
        roots = {item.pid for item in process_infos if item.role == "桌面程序"}
        for info in process_infos:
            progress(f"正在请求关闭 {info.name}（PID {info.pid}）…")
            time.sleep(0.10)
        self._post_close_to_windows(roots)
        _, alive = psutil.wait_procs(processes, timeout=4)

        # Children first; this prevents a model worker from surviving its UI.
        depths: dict[int, int] = {}
        by_pid = {item.pid: item for item in process_infos}
        for info in process_infos:
            depth = 0
            parent = info.ppid
            while parent in by_pid and depth < 16:
                depth += 1
                parent = by_pid[parent].ppid
            depths[info.pid] = depth
        alive.sort(key=lambda item: depths.get(item.pid, 0), reverse=True)
        stopped: list[int] = []
        for process in alive:
            try:
                progress(f"正在关闭 {process.name()}（PID {process.pid}）…")
                process.terminate()
                process.wait(timeout=2.5)
                stopped.append(process.pid)
            except psutil.TimeoutExpired:
                progress(f"{process.name()} 未响应，正在强制结束（PID {process.pid}）…")
                try:
                    process.kill()
                    process.wait(timeout=2)
                    stopped.append(process.pid)
                except psutil.Error:
                    pass
            except psutil.Error:
                continue
        stopped.extend(pid for pid in by_pid if not psutil.pid_exists(pid) and pid not in stopped)
        self.managed_pids.difference_update(stopped)
        if not self.list_processes():
            self.session_managed = False
        return stopped

    def restart(self, executable: str, install_dir: str, progress: StopCallback) -> int:
        self.stop_all(progress, managed_only=False)
        progress("VoiceStudio 后台已停止，正在重新启动桌面程序…")
        return self.start(executable, install_dir)

    def uninstall(
        self,
        progress: ProgressCallback,
        stop_callback: StopCallback,
        cancel_event: threading.Event | None = None,
    ) -> None:
        installation = self.detect_installation()
        if not installation.installed:
            raise VoiceStudioManagerError("没有检测到已安装的 VoiceStudio。")
        self.stop_all(stop_callback, managed_only=False)
        self._raise_if_cancelled(cancel_event)
        command = installation.quiet_uninstall_string or installation.uninstall_string
        product_code = _PRODUCT_CODE_RE.search(command)
        if product_code:
            arguments = ["msiexec.exe", "/x", product_code.group(0), "/passive", "/norestart"]
        elif command:
            arguments = ["cmd.exe", "/d", "/s", "/c", command]
        else:
            raise VoiceStudioManagerError(
                "没有找到 MSI 卸载信息。若这是便携版，请在 VoiceStudio 内备份数据后手动删除程序目录。"
            )
        self._run_installer(
            arguments,
            progress,
            "正在卸载 VoiceStudio（声音、模型和项目数据将保留）…",
            cancel_event,
        )
        progress("VoiceStudio 程序已卸载；用户数据仍保留。", 100)
