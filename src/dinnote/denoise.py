"""
Uses Demucs to isolate vocals in audio.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from .utils import cuda_available


def _ffmpeg_env():
    """Return env with FFmpeg bin dir on PATH so subprocess can load torchcodec DLLs."""
    import glob as g
    env = dict(os.environ)
    if sys.platform != "win32":
        return env
    local_appdata = env.get("LOCALAPPDATA", "")
    winget_packages = os.path.join(local_appdata, "Microsoft", "WinGet", "Packages")
    for bin_dir in g.glob(os.path.join(winget_packages, "Gyan.FFmpeg.Shared*", "**", "bin"), recursive=True):
        if any(g.glob(os.path.join(bin_dir, "avcodec-*.dll"))):
            env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
            break
    return env


def _run_demucs(input_file: Path, output_file: Path, model: str = "htdemucs", device: str = "cpu"):
    """Run Demucs and copy the vocals stem to the destination path."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        result = subprocess.run(
            [sys.executable, "-m", "demucs", "-n", model, "--two-stems=vocals",
             "--device", device, "-o", str(tmp_path), str(input_file)],
            env=_ffmpeg_env(),
        )
        if result.returncode != 0:
            raise RuntimeError(f"demucs exited with code {result.returncode}")

        vocals_path = tmp_path / model / input_file.stem / "vocals.wav"
        if not vocals_path.exists():
            raise FileNotFoundError(f"Demucs output not found: {vocals_path}")

        shutil.copy2(vocals_path, output_file)


def run(input_path: Path, output_dir: Path, config: dict, force: bool = False) -> Path:
    """Denoise a single audio file using Demucs vocal isolation. Returns path to denoised .wav."""
    output_file = output_dir / f"{output_dir.name}_denoised.wav"
    if not force and output_file.exists():
        return output_file

    model = config.get("model", "htdemucs")
    device = "cuda" if cuda_available() else "cpu"
    output_dir.mkdir(parents=True, exist_ok=True)
    _run_demucs(input_path, output_file, model=model, device=device)
    return output_file
