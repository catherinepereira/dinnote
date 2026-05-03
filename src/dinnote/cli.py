import argparse
import sys
import time
from pathlib import Path
from .utils import AUDIO_EXTENSIONS, load_config, warn_if_no_cuda, setup_user_config, fmt_time
from .pipeline import process_file
from .config import PipelineConfig


def main():
    parser = argparse.ArgumentParser(
        description="Denoise, VAD, diarize, and transcribe audio files"
    )
    parser.add_argument("input")
    parser.add_argument("-o", "--output", default="output")
    parser.add_argument("-c", "--config", default=None,
                        help="Path to config.yaml (default: user config dir)")
    parser.add_argument("-f", "--force", action="store_true")
    args = parser.parse_args()

    config_dir = setup_user_config()
    config_path = args.config or str(config_dir / "config.yaml")
    raw = load_config(config_path)

    transcribe_raw = raw.setdefault("transcribe", {})
    if not transcribe_raw.get("vocab_file"):
        vocab_path = config_dir / "vocab.txt"
        if vocab_path.exists():
            transcribe_raw["vocab_file"] = str(vocab_path)

    config = PipelineConfig.from_dict(raw)
    input_path = Path(args.input)
    output_dir = Path(args.output)

    if input_path.is_file():
        if input_path.suffix.lower() not in AUDIO_EXTENSIONS:
            print(f"Unrecognized audio extension: {input_path.suffix}")
            sys.exit(1)
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(
            f for f in input_path.iterdir()
            if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
        )
        if not files:
            print(f"No audio files found in {input_path}/")
            sys.exit(1)
    else:
        print(f"Not found: {input_path}")
        sys.exit(1)

    print(f"Files to process: {len(files)}")
    print(f"Output directory: {output_dir}/")
    warn_if_no_cuda()

    success = 0
    batch_start = time.monotonic()

    for i, audio_file in enumerate(files, 1):
        if len(files) > 1:
            print(f"\n{'═' * 60}")
            print(f"  File {i}/{len(files)}: {audio_file.name}")

        if process_file(audio_file, output_dir, config, force=args.force):
            success += 1

    failed = len(files) - success

    if len(files) > 1:
        print(f"\n{'─' * 60}")
        print(f"  Batch complete: {success} succeeded, {failed} failed")
        print(f"  Total time: {fmt_time(time.monotonic() - batch_start)}")
        print(f"\n{'─' * 60}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
