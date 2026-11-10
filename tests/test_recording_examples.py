"""The exported recording bundle is readable without the Rung installation."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


def load_inspector():
    path = Path(__file__).resolve().parents[1] / "examples" / "inspect_recording.py"
    spec = importlib.util.spec_from_file_location("recording_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_has_verifiable_original_bytes_and_refuses_overwrite_or_recursive_export(tmp_path):
    inspector = load_inspector()
    root = tmp_path / "recording"
    root.mkdir()
    (root / "trace.jsonl").write_text('{"kind":"observation"}\n')
    (root / "frame.png").write_bytes(b"example pixel bytes")
    destination = tmp_path / "episode.zip"
    inspector.export(root, destination)
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read("manifest.json"))
        assert len(manifest["files"]) == 2
        for entry in manifest["files"]:
            data = archive.read(entry["path"])
            assert len(data) == entry["bytes"]
            assert hashlib.sha256(data).hexdigest() == entry["sha256"]
    with pytest.raises(FileExistsError):
        inspector.export(root, destination)
    with pytest.raises(ValueError, match="outside"):
        inspector.export(root, root / "recursive.zip")
    assert (root / "trace.jsonl").read_text() == '{"kind":"observation"}\n'


def test_export_rejects_symlinks(tmp_path):
    root = tmp_path / "recording"
    root.mkdir()
    (root / "external").symlink_to(tmp_path / "private")
    with pytest.raises(ValueError, match="symlinks"):
        load_inspector().export(root, tmp_path / "episode.zip")


def test_documented_record_inspect_export_commands(tmp_path):
    examples = Path(__file__).resolve().parents[1] / "examples"
    recording = tmp_path / "episode"
    run = subprocess.run(
        [sys.executable, str(examples / "mock_episode.py"), "--out", str(recording)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "success" in run.stdout
    archive = tmp_path / "episode.zip"
    inspected = subprocess.run(
        [
            sys.executable,
            str(examples / "inspect_recording.py"),
            str(recording),
            "--export",
            str(archive),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert '"success": true' in inspected.stdout
    assert archive.is_file()


def test_remote_example_reports_missing_access_before_connecting(tmp_path):
    example = Path(__file__).resolve().parents[1] / "examples" / "remote_episode.py"
    env = {key: value for key, value in os.environ.items() if key != "RUNG_API_KEY"}
    result = subprocess.run(
        [sys.executable, str(example), "--url", "wss://example.invalid", "--out", str(tmp_path)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "set RUNG_API_KEY" in result.stderr
