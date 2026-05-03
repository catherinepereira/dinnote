"""
Transcribes speech segments from denoised audio using Whisper.

Accepts either:
  - a diarization JSON (preferred) — per-speaker turns from pyannote
  - a VAD JSON (fallback) — speaker-agnostic segments from Silero

When using diarization, each transcription entry includes the speaker label.
"""

import json
import shutil
from pathlib import Path
from typing import Optional, Callable
from pydub import AudioSegment
import whisper
from .utils import cuda_available

_whisper_cache: dict = {}


def _load_model(model_name: str, device: str):
    key = (model_name, device)
    if key not in _whisper_cache:
        _whisper_cache[key] = whisper.load_model(model_name, device=device)
    return _whisper_cache[key]


def _load_vocabulary(vocab_file: Optional[str]) -> str:
    """Build Whisper's initial_prompt from a vocabulary file."""
    if not vocab_file:
        return ""
    vocab_path = Path(vocab_file)
    if not vocab_path.exists():
        return ""
    terms = [
        line.strip()
        for line in vocab_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    return f"Common terms: {', '.join(terms)}." if terms else ""


def _load_segments(diarization_path: Optional[Path], vad_path: Optional[Path]):
    """Load segments from diarization JSON (preferred) or VAD JSON (fallback).

    Returns (segments, source) where each segment has start_ms, end_ms,
    and optionally speaker. source is "diarization" or "vad".
    """
    if diarization_path and diarization_path.exists():
        data = json.loads(diarization_path.read_text(encoding="utf-8"))
        turns = data.get("turns", [])
        segments = [
            {"start_ms": t["start_ms"], "end_ms": t["end_ms"], "speaker": t["speaker"]}
            for t in turns
        ]
        return segments, "diarization"

    if vad_path and vad_path.exists():
        data = json.loads(vad_path.read_text(encoding="utf-8"))
        segments = [
            {"start_ms": s["start_ms"], "end_ms": s["end_ms"]}
            for s in data.get("segments", [])
        ]
        return segments, "vad"

    raise RuntimeError("No diarization or VAD file provided.")


def run(
    audio_path: Path,
    output_dir: Path,
    config: dict,
    diarization_path: Optional[Path] = None,
    vad_path: Optional[Path] = None,
    on_segment: Optional[Callable] = None,
    force: bool = False,
) -> Path:
    """Transcribe speaker turns from denoised audio. Returns path to transcription JSON."""
    output_file = output_dir / f"{output_dir.name}_transcription.json"
    if not force and output_file.exists():
        return output_file

    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = config.get("model", "base")
    language = config.get("language", "en")
    temperature = config.get("temperature", None)
    no_speech_threshold = config.get("no_speech_threshold", None)
    logprob_threshold = config.get("logprob_threshold", None)
    compression_ratio_threshold = config.get("compression_ratio_threshold", None)
    condition_on_previous_text = config.get("condition_on_previous_text", False)
    vocab_file = config.get("vocab_file", "vocab.txt")

    device = "cuda" if cuda_available() else "cpu"
    model = _load_model(model_name, device)
    initial_prompt = _load_vocabulary(vocab_file) or None

    segments, source = _load_segments(diarization_path, vad_path)
    if not segments:
        raise RuntimeError("No segments found to transcribe.")

    audio = AudioSegment.from_file(str(audio_path))
    audio_duration_ms = len(audio)

    valid_segments = [seg for seg in segments if seg["start_ms"] < audio_duration_ms]
    for seg in valid_segments:
        seg["end_ms"] = min(seg["end_ms"], audio_duration_ms)

    total = len(valid_segments)

    output = {
        "metadata": {
            "source_audio": audio_path.name,
            "model": model_name,
            "language": language,
            "segment_source": source,
            "temperature": temperature,
            "no_speech_threshold": no_speech_threshold,
            "logprob_threshold": logprob_threshold,
            "compression_ratio_threshold": compression_ratio_threshold,
            "total_segments": total,
            "processed_segments": 0,
        },
        "transcription": [],
    }
    _write_json(output_file, output)

    temp_dir = output_dir / "_temp_segments"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()

    try:
        for i, seg in enumerate(valid_segments, 1):
            start_ms = seg["start_ms"]
            end_ms = seg["end_ms"]
            speaker = seg.get("speaker")

            temp_file = temp_dir / f"seg_{i}.wav"
            audio[start_ms:end_ms].export(str(temp_file), format="wav")

            try:
                transcribe_kwargs = dict(
                    language=language,
                    fp16=(device == "cuda"),
                    condition_on_previous_text=condition_on_previous_text,
                    initial_prompt=initial_prompt,
                )
                if temperature is not None:
                    transcribe_kwargs["temperature"] = temperature
                if no_speech_threshold is not None:
                    transcribe_kwargs["no_speech_threshold"] = no_speech_threshold
                if logprob_threshold is not None:
                    transcribe_kwargs["logprob_threshold"] = logprob_threshold
                if compression_ratio_threshold is not None:
                    transcribe_kwargs["compression_ratio_threshold"] = compression_ratio_threshold
                result = model.transcribe(str(temp_file), **transcribe_kwargs)
            except Exception:
                temp_file.unlink(missing_ok=True)
                if on_segment:
                    on_segment(i, total)
                continue

            temp_file.unlink(missing_ok=True)

            text = result["text"].strip()
            no_speech_prob = max(
                (s.get("no_speech_prob", 0) for s in result.get("segments", [])),
                default=0,
            )

            output["metadata"]["processed_segments"] += 1

            if not text or (no_speech_threshold is not None and no_speech_prob > no_speech_threshold):
                _write_json(output_file, output)
                if on_segment:
                    on_segment(i, total)
                continue

            entry = {
                "timestamp": {"start": start_ms / 1000, "end": end_ms / 1000},
                "text": text,
            }
            if speaker is not None:
                entry["speaker"] = speaker

            output["transcription"].append(entry)
            _write_json(output_file, output)

            if on_segment:
                on_segment(i, total)
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    return output_file


def _write_json(path: Path, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
