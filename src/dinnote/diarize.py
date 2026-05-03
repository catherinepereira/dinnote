"""
Uses pyannote speaker diarization to split denoised audio into per-speaker turns,
writing a diarization JSON that replaces VAD segments as the unit fed to Whisper.

Each turn is a single-speaker slice with absolute timestamps in the source audio,
so Whisper gets clean single-speaker audio and speaker attribution is built-in.

Output JSON format:
{
  "metadata": { ... },
  "turns": [
    {"turn_id": 0, "speaker": "SPEAKER_00", "start_ms": 1200, "end_ms": 3450, "duration_ms": 2250},
    ...
  ]
}
"""

import json
import os
import sys
from pathlib import Path
from .utils import cuda_available

_pipeline_cache = None


def _load_pipeline(hf_token: str, num_speakers: int | None, device: str):
    global _pipeline_cache
    if _pipeline_cache is None:
        from pyannote.audio import Pipeline
        import torch
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token,
        )
        pipeline.to(torch.device(device))
        _pipeline_cache = pipeline
    return _pipeline_cache


def _ensure_ffmpeg_dlls():
    """Register FFmpeg shared DLL directory on Windows (required by torchcodec)."""
    if sys.platform != "win32":
        return
    import glob as g
    local_appdata = os.environ.get("LOCALAPPDATA", "")
    winget_packages = os.path.join(local_appdata, "Microsoft", "WinGet", "Packages")
    if not os.path.isdir(winget_packages):
        return
    for bin_dir in g.glob(os.path.join(winget_packages, "Gyan.FFmpeg.Shared*", "**", "bin"), recursive=True):
        if any(g.glob(os.path.join(bin_dir, "avcodec-*.dll"))):
            os.add_dll_directory(bin_dir)
            return


def run(
    audio_path: Path,
    output_dir: Path,
    config: dict,
    force: bool = False,
) -> Path:
    """Run pyannote diarization on a denoised audio file. Returns path to diarization JSON."""
    output_file = output_dir / f"{output_dir.name}_diarization.json"
    if not force and output_file.exists():
        return output_file

    _ensure_ffmpeg_dlls()

    hf_token = config.get("hf_token")
    if not hf_token:
        raise RuntimeError(
            "HuggingFace token required for pyannote. "
            "Set diarize.hf_token in config.yaml."
        )

    num_speakers = config.get("num_speakers", None)
    min_speakers = config.get("min_speakers", None)
    max_speakers = config.get("max_speakers", None)
    min_turn_ms = config.get("min_turn_ms", 200)
    device = "cuda" if cuda_available() else "cpu"

    pipeline = _load_pipeline(hf_token, num_speakers, device)

    import torchaudio
    waveform, sr = torchaudio.load(str(audio_path))
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    diarize_kwargs = {}
    if num_speakers is not None:
        diarize_kwargs["num_speakers"] = num_speakers
    if min_speakers is not None:
        diarize_kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        diarize_kwargs["max_speakers"] = max_speakers

    result = pipeline({"waveform": waveform, "sample_rate": sr}, **diarize_kwargs)
    del waveform

    # Extract the Annotation object (handles pyannote API differences)
    annotation = result.speaker_diarization if hasattr(result, "speaker_diarization") else result

    speakers = sorted({s for _, _, s in annotation.itertracks(yield_label=True)})

    turns = []
    for i, (turn, _, speaker) in enumerate(annotation.itertracks(yield_label=True)):
        start_ms = round(turn.start * 1000)
        end_ms = round(turn.end * 1000)
        duration_ms = end_ms - start_ms
        if duration_ms < min_turn_ms:
            continue
        turns.append({
            "turn_id": i,
            "speaker": speaker,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_ms": duration_ms,
        })

    for i, t in enumerate(turns):
        t["turn_id"] = i

    from pydub import AudioSegment
    audio_duration_ms = len(AudioSegment.from_file(str(audio_path)))

    output_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "metadata": {
            "source_audio": audio_path.name,
            "audio_duration_ms": audio_duration_ms,
            "speakers_detected": speakers,
            "num_turns": len(turns),
            "total_speech_ms": sum(t["duration_ms"] for t in turns),
            "num_speakers_param": num_speakers,
            "min_turn_ms": min_turn_ms,
        },
        "turns": turns,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return output_file
