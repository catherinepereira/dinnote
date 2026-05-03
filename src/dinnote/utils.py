import os
import shutil
import sys
from importlib import resources
from pathlib import Path

AUDIO_EXTENSIONS = frozenset({".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"})

_CONFIG_FILES = ("config.yaml", "vocab.txt")


def get_config_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home()))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "dinnote"


def setup_user_config() -> Path:
    """Copy default config files to the user config dir if they don't exist yet."""
    config_dir = get_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)

    pkg = resources.files("dinnote")
    for name in _CONFIG_FILES:
        dest = config_dir / name
        if not dest.exists():
            src = pkg.joinpath(name)
            with resources.as_file(src) as src_path:
                shutil.copy2(src_path, dest)

    return config_dir


def load_config(config_path: str, section: str = "") -> dict:
    try:
        import yaml
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get(section, {}) if section else cfg
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def progress_bar(current: int, total: int, width: int = 28) -> str:
    if total == 0:
        return "░" * width
    filled = int(width * current / total)
    return "▓" * filled + "░" * (width - filled)


def fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{int(seconds // 60)}m {seconds % 60:.0f}s"


def cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def warn_if_no_cuda() -> None:
    if not cuda_available():
        print("WARNING: CUDA is not available! Audio processing will run on CPU, which will be much slower.")
