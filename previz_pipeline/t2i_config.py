"""Shared production T2I defaults, without importing GPU libraries."""
import os
from pathlib import Path

T2I_MODEL_ID = 'baidu/ERNIE-Image-Turbo'
DEFAULT_T2I_MODEL_PATH = os.environ.get(
    'T2I_MODEL_PATH', str(Path(__file__).resolve().parents[1] / 'hf_models' / 'ERNIE-Image-Turbo'))
DEFAULT_T2I_STEPS = 8
DEFAULT_T2I_GUIDANCE = 1.0
