"""
Extends the dinscribe pipeline with a speaker diarization step (pyannote) between VAD and
transcription. Diarization replaces VAD segments as the unit fed to Whisper, giving each
transcription entry a speaker label and cleaner single-speaker audio.

Pipeline: denoise -> vad -> diarize -> transcribe
"""

import json
import time
from pathlib import Path

from . import denoise, vad, diarize, transcribe
from .config import PipelineConfig
from .utils import fmt_time, progress_bar, cuda_available


def process_file(input_path: Path, output_dir: Path, config: PipelineConfig = PipelineConfig(), force: bool = False) -> bool:
    """Run the full pipeline for one audio file. Returns True on success."""
    file_dir = output_dir / input_path.stem
    file_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.monotonic()

    print(f"\n{'─' * 60}")
    print(f"  Input: {input_path}")
    print(f"  Output: {file_dir}/")
    print('─' * 60)

    print("\n  [1/4] Denoising")
    denoised_path = file_dir / f"{input_path.stem}_denoised.wav"
    if not force and denoised_path.exists():
        print("        ✓ Skipped (cached)")
    else:
        print("        Running Demucs vocal isolation...")
        step_start = time.monotonic()
        try:
            import dataclasses
            denoised_path = denoise.run(input_path, file_dir, dataclasses.asdict(config.denoise), force=force)
        except Exception as e:
            print(f"\n  ERROR: Denoising failed: {e}")
            return False
        print(f"        ✓ Done  ({fmt_time(time.monotonic() - step_start)})")

    print("\n  [2/4] Voice Activity Detection")
    vad_path = file_dir / f"{input_path.stem}_vad.json"
    vad_cached = not force and vad_path.exists()
    step_start = time.monotonic()
    if vad_cached:
        print("        ✓ Skipped (cached)")
    else:
        print("        Detecting speech segments...")
        try:
            import dataclasses
            vad_path = vad.run(denoised_path, file_dir, dataclasses.asdict(config.vad), force=force)
        except Exception as e:
            print(f"\n  ERROR: VAD failed: {e}")
            return False

    vad_meta = json.loads(vad_path.read_text(encoding="utf-8"))["metadata"]
    if not vad_cached:
        speech_s = vad_meta["total_speech_ms"] / 1000
        audio_s = vad_meta["audio_duration_ms"] / 1000
        print(f"        ✓ {vad_meta['segment_count']} segments found  "
              f"({speech_s:.1f}s speech in {audio_s:.1f}s audio)  "
              f"({fmt_time(time.monotonic() - step_start)})")

    print("\n  [3/4] Speaker Diarization")
    diarization_path = file_dir / f"{input_path.stem}_diarization.json"
    diar_cached = not force and diarization_path.exists()
    step_start = time.monotonic()
    if diar_cached:
        print("        ✓ Skipped (cached)")
    else:
        import dataclasses
        diar_config = dataclasses.asdict(config.diarize)
        num_spk = diar_config.get("num_speakers")
        print(f"        Running pyannote{f' (num_speakers={num_spk})' if num_spk else ''}...")
        try:
            diarization_path = diarize.run(denoised_path, file_dir, diar_config, force=force)
        except Exception as e:
            print(f"\n  ERROR: Diarization failed: {e}")
            return False

    diar_meta = json.loads(diarization_path.read_text(encoding="utf-8"))["metadata"]
    if not diar_cached:
        speakers = diar_meta["speakers_detected"]
        print(f"        ✓ {len(speakers)} speaker(s): {', '.join(speakers)}  "
              f"| {diar_meta['num_turns']} turns  "
              f"({fmt_time(time.monotonic() - step_start)})")

    print("\n  [4/4] Transcribing")
    transcription_path = file_dir / f"{input_path.stem}_transcription.json"
    trans_cached = not force and transcription_path.exists()
    if trans_cached:
        print("        ✓ Skipped (cached)")
    else:
        import dataclasses
        trans_config = dataclasses.asdict(config.transcribe)
        model_name = trans_config.get("model", "base")
        device = "cuda" if cuda_available() else "cpu"
        print(f"        Loading Whisper '{model_name}' on {device}...")
        step_start = time.monotonic()

        def on_segment(current: int, total: int):
            bar = progress_bar(current, total)
            print(f"\r        [{bar}] {current}/{total} turns", end="", flush=True)

        try:
            transcription_path = transcribe.run(
                denoised_path,
                file_dir,
                trans_config,
                diarization_path=diarization_path,
                vad_path=vad_path,
                on_segment=on_segment,
                force=force,
            )
        except Exception as e:
            print(f"\n  ERROR: Transcription failed: {e}")
            return False

        print()
        trans_meta = json.loads(transcription_path.read_text(encoding="utf-8"))["metadata"]
        kept = trans_meta["processed_segments"]
        total_turns = diar_meta["num_turns"]
        print(f"        ✓ {kept}/{total_turns} turns transcribed  "
              f"({fmt_time(time.monotonic() - step_start)})")

    print(f"\n{'─' * 60}")
    print(f"  Total time: {fmt_time(time.monotonic() - total_start)}")
    print("  Output:")
    for p in (denoised_path, vad_path, diarization_path, transcription_path):
        print(f"    {p.relative_to(output_dir.parent)}")
    print('─' * 60)

    return True
