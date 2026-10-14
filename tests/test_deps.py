"""PDF 组件（pypdf）检测与在线安装：全部离线，pip 用假的 runner 代替。"""
from __future__ import annotations

import os
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYNPUT_BACKEND", "dummy")

from teams_voice_translator.defense import deps, thesis_context

ROOT = Path(__file__).resolve().parents[1]


def _module(version: str):
    return types.SimpleNamespace(__version__=version, PdfReader=object)


def _importer(result):
    def importer(name):
        if isinstance(result, Exception):
            raise result
        return result
    return importer


class StatusTests(unittest.TestCase):
    def test_missing(self):
        self.assertEqual(deps.pypdf_status(_importer(ImportError("x"))).state, "missing")

    def test_old_version(self):
        status = deps.pypdf_status(_importer(_module("3.17.4")))
        self.assertEqual((status.state, status.version, status.ok), ("old", "3.17.4", False))

    def test_new_version(self):
        self.assertTrue(deps.pypdf_status(_importer(_module("6.1.0"))).ok)

    def test_garbage_version_counts_as_old(self):
        self.assertEqual(deps.pypdf_status(_importer(_module(""))).state, "old")


class InstallTests(unittest.TestCase):
    def test_command_uses_current_interpreter_and_binary_only(self):
        command = deps.install_command("pypdf>=5,<7", deps.MIRRORS[0], proxy="http://127.0.0.1:7890")
        self.assertEqual(command[:4], [sys.executable, "-m", "pip", "install"])
        self.assertIn("--only-binary=:all:", command)
        self.assertEqual(command[-2:], ["--proxy", "http://127.0.0.1:7890"])
        self.assertNotIn("--proxy", deps.install_command("pypdf", deps.MIRRORS[0]))

    def test_first_mirror_success(self):
        calls = []

        def runner(command, **kwargs):
            calls.append(command)
            self.assertIn("timeout", kwargs)
            return subprocess.CompletedProcess(command, 0, "Successfully installed pypdf-6.1.0", "")

        with mock.patch.object(deps, "is_frozen", return_value=False):
            ok, output = deps.install_package(runner=runner)
        self.assertTrue(ok)
        self.assertIn("Successfully", output)
        self.assertEqual(len(calls), 1)
        self.assertIn("tuna.tsinghua", " ".join(calls[0]))

    def test_falls_back_to_official_index(self):
        messages = []
        results = iter([
            subprocess.TimeoutExpired("pip", 240),
            subprocess.CompletedProcess("pip", 0, "ok", ""),
        ])

        def runner(command, **kwargs):
            result = next(results)
            if isinstance(result, Exception):
                raise result
            return result

        with mock.patch.object(deps, "is_frozen", return_value=False):
            ok, _output = deps.install_package(runner=runner, on_status=messages.append)
        self.assertTrue(ok)
        self.assertEqual(len(messages), 2)
        self.assertIn("pypi.org", messages[1])

    def test_all_mirrors_fail(self):
        def runner(command, **kwargs):
            return subprocess.CompletedProcess(command, 1, "", "ERROR: No matching distribution")

        with mock.patch.object(deps, "is_frozen", return_value=False):
            ok, output = deps.install_package(runner=runner)
        self.assertFalse(ok)
        self.assertIn("No matching distribution", output)

    def test_frozen_exe_never_runs_pip(self):
        def runner(command, **kwargs):  # pragma: no cover - 不应被调用
            raise AssertionError("EXE 里不能调用 pip")

        with mock.patch.object(deps, "is_frozen", return_value=True):
            self.assertFalse(deps.can_auto_install())
            ok, _ = deps.install_package(runner=runner)
        self.assertFalse(ok)


class LoadTests(unittest.TestCase):
    def test_old_module_already_loaded_needs_restart(self):
        with mock.patch.dict(sys.modules, {"pypdf": _module("3.17.0")}):
            self.assertIsNone(deps.load_pypdf())

    def test_new_module_already_loaded(self):
        module = _module("6.0.0")
        with mock.patch.dict(sys.modules, {"pypdf": module}):
            self.assertIs(deps.load_pypdf(), module)

    def test_thesis_context_picks_up_freshly_installed_module(self):
        module = _module("6.0.0")
        with mock.patch.object(thesis_context, "pypdf", None), \
                mock.patch.object(deps, "load_pypdf", return_value=module):
            self.assertIs(thesis_context._pypdf(), module)

    def test_missing_module_gives_friendly_error(self):
        with mock.patch.object(thesis_context, "pypdf", None), \
                mock.patch.object(deps, "load_pypdf", return_value=None):
            with self.assertRaises(thesis_context.ApiError) as ctx:
                thesis_context.extract_pdf_pages(Path("x.pdf"))
        self.assertIn("Word", str(ctx.exception))


class BuildFileTests(unittest.TestCase):
    def test_requirements_pin_new_pypdf(self):
        self.assertIn("pypdf>=5", (ROOT / "requirements.txt").read_text(encoding="utf-8"))

    def test_spec_bundles_pypdf(self):
        spec = ROOT / "DefenseMode.spec"
        if not spec.exists():
            self.skipTest("打包文件不在测试副本里")
        self.assertIn("collect_all('pypdf')", spec.read_text(encoding="utf-8"))

    def test_build_script_installs_requirements_first(self):
        script = ROOT / "build_defense.ps1"
        if not script.exists():
            self.skipTest("打包脚本不在测试副本里")
        text = script.read_text(encoding="utf-8")
        self.assertLess(text.index("requirements-build.txt"), text.index("PyInstaller --noconfirm"))
        self.assertIn("import pypdf", text)


class WindowFlowTests(unittest.TestCase):
    """界面流程：缺组件 → 询问 → 后台安装 → 自动接着导入。"""

    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def _window(self):
        from teams_voice_translator.defense import ui

        window = types.SimpleNamespace()
        window.ppt_button = mock.Mock()
        bar = mock.Mock()
        window.statusBar = lambda: bar
        window.settings = types.SimpleNamespace(proxy="http://p:1", proxy_mode="manual")
        window.signals = mock.Mock()
        window.imported = []
        window._import_thesis = window.imported.append
        for name in ("_ensure_pdf_support", "_install_pdf_support", "_on_pdf_support_ready"):
            setattr(window, name, types.MethodType(getattr(ui.DefenseWindow, name), window))
        return ui, window

    def test_ok_status_imports_directly(self):
        ui, window = self._window()
        with mock.patch.object(deps, "pypdf_status", return_value=deps.DependencyStatus("ok", "6.0")):
            self.assertTrue(window._ensure_pdf_support("a.pdf"))

    def test_user_declines(self):
        ui, window = self._window()
        with mock.patch.object(deps, "pypdf_status", return_value=deps.DependencyStatus("missing")), \
                mock.patch.object(deps, "can_auto_install", return_value=True), \
                mock.patch.object(ui.QMessageBox, "question", return_value=ui.QMessageBox.No), \
                mock.patch.object(deps, "install_package") as install:
            self.assertFalse(window._ensure_pdf_support("a.pdf"))
        install.assert_not_called()

    def test_accept_installs_then_continues_import(self):
        ui, window = self._window()
        with mock.patch.object(deps, "pypdf_status", return_value=deps.DependencyStatus("missing")), \
                mock.patch.object(deps, "can_auto_install", return_value=True), \
                mock.patch.object(ui.QMessageBox, "question", return_value=ui.QMessageBox.Yes), \
                mock.patch.object(deps, "install_package", return_value=(True, "done")) as install:
            self.assertFalse(window._ensure_pdf_support("a.pdf"))
            window.pdf_install_worker.join(5)
        self.assertEqual(install.call_args.kwargs["proxy"], "http://p:1")
        window.ppt_button.setEnabled.assert_called_with(False)
        window.signals.pdf_support_ready.emit.assert_called_once_with("a.pdf", True, "done")
        with mock.patch.object(deps, "load_pypdf", return_value=_module("6.0.0")):
            window._on_pdf_support_ready("a.pdf", True, "done")
        self.assertEqual(window.imported, ["a.pdf"])
        window.ppt_button.setEnabled.assert_called_with(True)

    def test_install_failure_shows_warning_and_does_not_import(self):
        ui, window = self._window()
        with mock.patch.object(ui.QMessageBox, "warning") as warning:
            window._on_pdf_support_ready("a.pdf", False, "line1\nERROR: network down")
        self.assertIn("network down", warning.call_args.args[2])
        self.assertEqual(window.imported, [])

    def test_old_version_loaded_asks_for_restart(self):
        ui, window = self._window()
        with mock.patch.object(deps, "load_pypdf", return_value=None), \
                mock.patch.object(ui.QMessageBox, "information") as info:
            window._on_pdf_support_ready("a.pdf", True, "")
        self.assertIn("重新打开", info.call_args.args[2])
        self.assertEqual(window.imported, [])

    def test_frozen_exe_missing_gives_guidance(self):
        ui, window = self._window()
        with mock.patch.object(deps, "pypdf_status", return_value=deps.DependencyStatus("missing")), \
                mock.patch.object(deps, "can_auto_install", return_value=False), \
                mock.patch.object(ui.QMessageBox, "warning") as warning:
            self.assertFalse(window._ensure_pdf_support("a.pdf"))
        self.assertIn("Word", warning.call_args.args[2])

    def test_old_version_can_be_ignored(self):
        ui, window = self._window()
        with mock.patch.object(deps, "pypdf_status", return_value=deps.DependencyStatus("old", "3.17")), \
                mock.patch.object(deps, "can_auto_install", return_value=True), \
                mock.patch.object(ui.QMessageBox, "question", return_value=ui.QMessageBox.Ignore):
            self.assertTrue(window._ensure_pdf_support("a.pdf"))


if __name__ == "__main__":
    unittest.main()
