"""Runtime locations; no machine-specific paths or credentials."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get('TUNBAO_DATA_ROOT', REPO_ROOT / 'data')).expanduser().resolve()
T2M_ROOT = Path(os.environ.get('T2M_GPT_ROOT', REPO_ROOT / 'vendor/t2m_gpt')).expanduser().resolve()
CLIP_PATH = Path(os.environ.get('CLIP_MODEL_PATH', Path.home() / '.cache/clip/ViT-B-32.pt')).expanduser()
