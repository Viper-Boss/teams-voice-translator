from __future__ import annotations

import threading
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from teams_voice_translator.voicestudio import (
    VoiceStudioClient,
    normalize_voicestudio_url,
    parse_voice_catalog,
)
from teams_voice_translator.voicestudio_manager import (
    VoiceStudioInstallation,
    VoiceStudioManager,
    VoiceStudioTaskCancelled,
    parse_release_payload,
)


class VoiceStudioTests(unittest.TestCase):
    def test_normalize_base_url(self) -> None:
        self.assertEqual(normalize_voicestudio_url("localhost:3900/v1/"), "http://localhost:3900")
        self.assertEqual(normalize_voicestudio_url("https://speech.example/v1"), "https://speech.example")

    def test_catalog_accepts_current_discovery_shape(self) -> None:
        voices, engines = parse_voice_catalog(
            {
                "voices": [
                    {
                        "voice_id": "profile-1",
                        "name": "My voice",
                        "type": "profile",
                        "language": "en",
                        "engine": "voxcpm2",
                    }
                ],
                "engines": [{"id": "voxcpm2", "name": "VoxCPM2", "installed": True}],
            }
        )
        self.assertEqual(voices[0].voice_id, "profile-1")
        self.assertEqual(voices[0].engine, "voxcpm2")
        self.assertEqual(engines[0].engine_id, "voxcpm2")
        self.assertTrue(engines[0].installed)

    def test_stream_tts_uses_local_openai_contract(self) -> None:
        client = VoiceStudioClient("http://127.0.0.1:3900")
        response = Mock()
        response.ok = True
        response.headers = {"Content-Type": "audio/pcm"}
        response.iter_content.return_value = [b"\x01\x00" * 10]
        client.session.post = Mock(return_value=response)
        parts: list[bytes] = []

        count = client.stream_tts(
            "Hello",
            {"voicestudio_model": "tts-1", "voicestudio_voice": "profile-1", "tts_rate": 1.15},
            parts.append,
            threading.Event(),
        )

        self.assertEqual(count, 1)
        self.assertEqual(parts, [b"\x01\x00" * 10])
        payload = client.session.post.call_args.kwargs["json"]
        self.assertEqual(payload["response_format"], "pcm")
        self.assertEqual(payload["voice"], "profile-1")
        self.assertEqual(payload["speed"], 1.15)

    def test_latest_release_extracts_windows_msi_and_github_sha256(self) -> None:
        checksum = "a" * 64
        release = parse_release_payload(
            {
                "tag_name": "v0.5.1",
                "html_url": "https://github.com/debpalash/VoiceStudio/releases/tag/v0.5.1",
                "body": f"### Windows x64 artifacts\n{checksum} *VoiceStudio_0.5.1_x64_en-US.msi\n",
                "assets": [
                    {
                        "name": "VoiceStudio_0.5.1_x64_en-US.msi",
                        "browser_download_url": "https://example.invalid/VoiceStudio.msi",
                        "size": 123,
                    }
                ],
            }
        )
        self.assertEqual(release.version, "0.5.1")
        self.assertEqual(release.expected_sha256, checksum)
        self.assertTrue(release.asset_name.endswith(".msi"))

    def test_installer_sha256_reports_progress(self) -> None:
        payload = b"voice-studio-installer" * 100
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "VoiceStudio.msi"
            path.write_bytes(payload)
            progress: list[tuple[str, int]] = []
            actual = VoiceStudioManager.file_sha256(
                path,
                lambda message, percent: progress.append((message, percent)),
            )
        self.assertEqual(actual, hashlib.sha256(payload).hexdigest())
        self.assertTrue(progress)
        self.assertEqual(progress[-1][1], 100)

    def test_sha256_can_be_cancelled(self) -> None:
        payload = b"voice-studio-installer" * 100
        cancel_event = threading.Event()
        cancel_event.set()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "VoiceStudio.msi"
            path.write_bytes(payload)
            with self.assertRaises(VoiceStudioTaskCancelled):
                VoiceStudioManager.file_sha256(path, cancel_event=cancel_event)

    def test_running_msi_process_is_terminated_when_cancelled(self) -> None:
        cancel_event = threading.Event()
        process = Mock()

        def poll() -> None:
            cancel_event.set()
            return None

        process.poll.side_effect = poll
        process.wait.return_value = 0
        with patch(
            "teams_voice_translator.voicestudio_manager.subprocess.Popen",
            return_value=process,
        ):
            with self.assertRaises(VoiceStudioTaskCancelled):
                VoiceStudioManager._run_installer(
                    ["msiexec.exe", "/i", "VoiceStudio.msi"],
                    lambda _message, _percent: None,
                    "正在安装…",
                    cancel_event,
                )
        process.terminate.assert_called_once()

    def test_detects_current_omnivoice_desktop_executable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "omnivoice-studio.exe"
            executable.write_bytes(b"test")
            registered = VoiceStudioInstallation(
                installed=True,
                version="0.5.1",
                install_dir=directory,
                display_name="VoiceStudio",
            )
            manager = VoiceStudioManager()
            with patch.object(manager, "_registry_installations", return_value=[registered]):
                detected = manager.detect_installation()
        self.assertTrue(detected.installed)
        self.assertEqual(detected.version, "0.5.1")
        self.assertEqual(Path(detected.executable).name, "omnivoice-studio.exe")


if __name__ == "__main__":
    unittest.main()
