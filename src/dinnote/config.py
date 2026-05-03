from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DenoiseConfig:
    model: str = "htdemucs"


@dataclass
class VadConfig:
    threshold: float = 0.5
    min_speech_duration_ms: Optional[int] = 250
    min_silence_duration_ms: Optional[int] = 100
    padding_ms: int = 500
    max_segment_length_sec: Optional[float] = 30.0
    """Segments longer than this are discarded as likely noise or music. None keeps all."""
    merge_within_sec: Optional[float] = 1.0
    """Segments with less than this gap between them are merged. None disables merging."""


@dataclass
class DiarizeConfig:
    hf_token: Optional[str] = None
    num_speakers: Optional[int] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    min_turn_ms: int = 200
    """Turns less than this length are discarded."""


@dataclass
class TranscribeConfig:
    model: str = "base"
    language: str = "en"
    temperature: Optional[float] = None
    """0 for greedy decoding, None to use Whisper's fallback sequence."""
    no_speech_threshold: Optional[float] = 0.6
    logprob_threshold: Optional[float] = -1.0
    compression_ratio_threshold: Optional[float] = 2.4
    condition_on_previous_text: bool = False
    """Use previous segment output as context. Improves coherence but can propagate errors."""
    vocab_file: Optional[str] = None


@dataclass
class PipelineConfig:
    denoise: DenoiseConfig = field(default_factory=DenoiseConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    diarize: DiarizeConfig = field(default_factory=DiarizeConfig)
    transcribe: TranscribeConfig = field(default_factory=TranscribeConfig)

    @staticmethod
    def from_dict(d: dict) -> "PipelineConfig":
        def _build(cls, section):
            fields = {f.name for f in cls.__dataclass_fields__.values()}
            return cls(**{k: v for k, v in section.items() if k in fields})

        return PipelineConfig(
            denoise=_build(DenoiseConfig, d.get("denoise", {})),
            vad=_build(VadConfig, d.get("vad", {})),
            diarize=_build(DiarizeConfig, d.get("diarize", {})),
            transcribe=_build(TranscribeConfig, d.get("transcribe", {})),
        )
