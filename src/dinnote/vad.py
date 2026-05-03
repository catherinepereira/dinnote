"""
Uses Silero VAD to detect speech segments in denoised audio and writes a JSON file with segment boundaries.
"""

import json
from pathlib import Path
import torch
from .utils import cuda_available
from .config import VadConfig

_vad_model_cache = None


def _load_model():
    global _vad_model_cache
    if _vad_model_cache is None:
        torch.set_num_threads(1)
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        get_speech_timestamps, _, read_audio, *_ = utils
        device = "cuda" if cuda_available() else "cpu"
        model = model.to(device)
        _vad_model_cache = (model, get_speech_timestamps, read_audio, device)
    return _vad_model_cache


def _detect_segments(audio_path: Path, config: dict):
    """Run Silero VAD and return (segments, duration_ms).

    segments is a list of [start_ms, end_ms] pairs.
    """
    threshold = config.get("threshold", 0.5)
    min_speech_ms = config.get("min_speech_duration_ms", None)
    min_silence_ms = config.get("min_silence_duration_ms", None)
    padding_ms = config.get("padding_ms", 500) or 0
    max_seg_sec = config.get("max_segment_length_sec", 30)
    merge_within_sec = config.get("merge_within_sec", 1.0)
    sample_rate = 16000  # required by Silero VAD

    model, get_speech_timestamps, read_audio, device = _load_model()

    wav = read_audio(str(audio_path), sampling_rate=sample_rate).to(device)
    duration_ms = len(wav) / sample_rate * 1000

    vad_kwargs = dict(
        threshold=threshold,
        sampling_rate=sample_rate,
        window_size_samples=512,
        speech_pad_ms=30,
    )
    if min_speech_ms is not None:
        vad_kwargs["min_speech_duration_ms"] = min_speech_ms
    if min_silence_ms is not None:
        vad_kwargs["min_silence_duration_ms"] = min_silence_ms
    speech_timestamps = get_speech_timestamps(wav, model, **vad_kwargs)

    # Convert sample indices to ms and apply padding
    samples_per_ms = sample_rate / 1000
    segments = []
    for ts in speech_timestamps:
        start_ms = max(0, int(ts["start"] / samples_per_ms) - padding_ms)
        end_ms = min(duration_ms, int(ts["end"] / samples_per_ms) + padding_ms)
        if segments and start_ms < segments[-1][1]:
            start_ms = segments[-1][1]
        if start_ms < end_ms:
            segments.append([start_ms, end_ms])

    # Merge nearby segments
    if merge_within_sec is not None:
        merge_gap_ms = merge_within_sec * 1000
        merged = []
        for start_ms, end_ms in segments:
            if merged and start_ms - merged[-1][1] <= merge_gap_ms:
                merged[-1][1] = end_ms
            else:
                merged.append([start_ms, end_ms])
        segments = merged

    # Discard segments that exceed the maximum length
    if max_seg_sec is not None:
        max_ms = max_seg_sec * 1000
        segments = [[s, e] for s, e in segments if e - s <= max_ms]

    return segments, duration_ms


def _build_output(audio_path: Path, segments: list, duration_ms: float, config: dict) -> dict:
    total_speech_ms = sum(e - s for s, e in segments)
    return {
        "metadata": {
            "source_audio": audio_path.name,
            "audio_duration_ms": duration_ms,
            "total_speech_ms": total_speech_ms,
            "segment_count": len(segments),
            "vad_threshold": config.get("threshold", 0.5),
            "max_segment_length_sec": config.get("max_segment_length_sec", 30),
            "merge_within_sec": config.get("merge_within_sec", 1.0),
        },
        "segments": [
            {"segment_id": i, "start_ms": s, "end_ms": e, "duration_ms": e - s}
            for i, (s, e) in enumerate(segments)
        ],
    }


def run(input_path: Path, output_dir: Path, config: dict, force: bool = False) -> Path:
    """Run Silero VAD on a single audio file. Returns path to vad.json."""
    output_file = output_dir / f"{output_dir.name}_vad.json"
    if not force and output_file.exists():
        return output_file

    output_dir.mkdir(parents=True, exist_ok=True)
    segments, duration_ms = _detect_segments(input_path, config)
    if not segments:
        raise RuntimeError(f"No speech segments detected in {input_path.name}")

    output = _build_output(input_path, segments, duration_ms, config)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return output_file
