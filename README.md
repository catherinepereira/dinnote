## dinnote audio transcription
Processes audio through a four-step pipeline to produce a transcription JSON with per-speaker diarization: denoising (Demucs), voice activity detection (Silero VAD), speaker diarization (pyannote), and transcription (Whisper).

### Installation
```bash
pip install dinnote
```

On first run, dinnote copies default config files to your platform config directory:
- **Windows:** `%APPDATA%\dinnote\`
- **macOS:** `~/Library/Application Support/dinnote/`
- **Linux:** `~/.config/dinnote/`

Edit `config.yaml` and `vocab.txt` to customize settings.

Speaker diarization requires a HuggingFace token with access to [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1). 
Set it via `diarize.hf_token` in `config.yaml`.


### CLI usage
```bash
dinnote input/audio.mp3            # single file
dinnote input/                     # all audio files in a folder
dinnote input/audio.mp3 -f         # force re-run all steps
dinnote input/audio.mp3 -c path/to/config.yaml   # custom config
dinnote input/audio.mp3 -o results/              # custom output dir
```

Each step checks whether its output already exists and skips it if so. Use `-f` to force all steps to re-run.

Output is written to `output/<filename>/` and contains:
- `<filename>_denoised.wav` (vocals isolated from background noise)
- `<filename>_vad.json` (detected speech segment boundaries)
- `<filename>_diarization.json` (per-speaker turn boundaries from pyannote)
- `<filename>_transcription.json` (final transcription with timestamps and speaker labels)


### Python API
```python
from pathlib import Path
import dinnote
from dinnote import PipelineConfig, VadConfig, DiarizeConfig, TranscribeConfig

# Run the full pipeline with defaults
dinnote.process_file(
    input_path=Path("recording.wav"),
    output_dir=Path("output"),
)

# Custom config
config = PipelineConfig(
    vad=VadConfig(threshold=0.4, max_segment_length_sec=20),
    diarize=DiarizeConfig(num_speakers=2),
    transcribe=TranscribeConfig(model="small", language="en"),
)
dinnote.process_file(Path("recording.wav"), Path("output"), config=config)

# Or use individual stages
from dinnote import denoise, vad, diarize, transcribe

denoised      = denoise.run(Path("recording.wav"), Path("output/recording"), config={})
vad_file      = vad.run(denoised, Path("output/recording"), config={})
diarization   = diarize.run(denoised, Path("output/recording"), config={})
result        = transcribe.run(denoised, Path("output/recording"), config={}, diarization_path=diarization)
```


### Configuration

```yaml
denoise:
  model: htdemucs        # htdemucs | htdemucs_ft | mdx | mdx_extra | htdemucs_6s

vad:
  threshold: 0.5         # 0.0–1.0, higher = requires clearer speech
  min_speech_duration_ms: 250
  min_silence_duration_ms: 100
  padding_ms: 500
  max_segment_length_sec: 30
  merge_within_sec: 1.0

diarize:
  # hf_token: hf_...
  num_speakers: null     # fix speaker count or leave null to let pyannote estimate
  min_speakers: null
  max_speakers: null
  min_turn_ms: 200       # turns shorter than this are discarded (ms)

transcribe:
  model: base            # tiny | base | small | medium | large
  language: en           # set to null to auto-detect
  temperature: null      # null = Whisper fallback sequence, 0 = greedy
  no_speech_threshold: 0.6
  logprob_threshold: -1.0
  compression_ratio_threshold: 2.4
  condition_on_previous_text: false
  vocab_file: null       # path to domain-specific vocabulary, defaults to vocab.txt in config dir
```

Add domain-specific vocabulary to `vocab.txt` to improve transcription accuracy on unusual words and jargon. For noisy or technical audio, set `temperature: 0` to disable Whisper's fallback to higher-temperature decoding, and consider filtering out common hallucinations specific to your dataset.

If `num_speakers` is known in advance, setting it gives more reliable diarization. Otherwise use `min_speakers`/`max_speakers` to constrain the range, or leave both null to let pyannote estimate freely.
