"""Inspect a local episode and optionally export a ZIP with SHA-256 checksums.

Works with the recorded mock and remote examples. Never modifies the source recording.
The ZIP contains all files under the supplied directory; review it before sharing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def inspect(root: Path) -> dict:
    events = [
        json.loads(line) for line in (root / "trace.jsonl").read_text().splitlines() if line.strip()
    ]
    episodes = []
    for directory in sorted((root / "episodes").iterdir()):
        if not directory.is_dir():
            continue
        result_path = directory / "result.json"
        interrupted = directory / "interruption.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else None
        server_path = directory / "server.json"
        episode_events = [event for event in events if event["episode"] == directory.name]
        episodes.append(
            {
                "episode": directory.name,
                "observations": len(list(directory.glob("obs_*.json"))),
                "rgb_frames": len(list(directory.glob("obs_*.png"))),
                "targets_requested": sum(event["kind"] == "attempt" for event in episode_events),
                "execution_reports": sum(event["kind"] == "execution" for event in episode_events),
                "result": result,
                "server": json.loads(server_path.read_text()) if server_path.exists() else None,
                "interrupted": json.loads(interrupted.read_text())
                if interrupted.exists()
                else None,
            }
        )
    return {"episodes": episodes, "trace_events": len(events)}


def export(root: Path, destination: Path) -> None:
    root = root.resolve()
    destination = destination.resolve()
    if destination == root or root in destination.parents:
        raise ValueError("archive destination must be outside the recording directory")
    paths = sorted(root.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("recording contains symlinks; export regular recording files only")
    files = [path for path in paths if path.is_file()]
    manifest = {"format": "rung-recording-bundle/1", "files": []}
    # Exclusive creation avoids overwriting an earlier archive.
    with destination.open("xb") as output:
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                data = path.read_bytes()
                relative = path.relative_to(root).as_posix()
                archive.writestr("recording/" + relative, data)
                manifest["files"].append(
                    {
                        "path": "recording/" + relative,
                        "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                )
            archive.writestr("manifest.json", json.dumps(manifest, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--export", type=Path, dest="archive", help="new ZIP outside the recording")
    args = parser.parse_args()
    try:
        print(json.dumps(inspect(args.recording), indent=2))
        if args.archive:
            export(args.recording, args.archive)
            print(f"Exported {args.archive}")
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Cannot inspect/export recording: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
