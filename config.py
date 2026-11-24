"""
Unified Configuration Module for GenAI Jury 2
Configured and optimized for NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB VRAM).
"""

import os
from pathlib import Path
import torch

# Base directories
BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"
CHECKPOINTS_DIR = BASE_DIR / "intent-classifier-checkpoints"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 1. Device and Hardware Optimization (RTX 4050 Laptop GPU)
# ---------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
TOTAL_VRAM_GB = (
    torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if torch.cuda.is_available()
    else 0.0
)

# RTX 4050 is Ada Lovelace (SM 8.9), natively supporting BF16 and FP16 with Tensor Cores
USE_FP16 = torch.cuda.is_available()
USE_BF16 = False  # Set to True if PyTorch CUDA build supports BF16 on current driver

# Memory management settings for 6GB VRAM
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = 32
MAX_SEQ_LENGTH = 128
DATALOADER_NUM_WORKERS = 0  # 0 is recommended on Windows to avoid process spawn overhead

# ---------------------------------------------------------
# 2. Dataset Selection & Scaling
# ---------------------------------------------------------
# Choices: "banking200k" (Massive 220,000 rows, 110 Banking Intents),
#          "massive" (Amazon MASSIVE: 16.5k rows, 60 intents),
#          "banking77" (PolyAI: 13.1k rows, 77 intents),
#          "merged" (Unified Multi-Domain Corpus: 29.6k rows, 137 intents)
DATASET_CHOICE = "banking200k"
SEED = 42

DATASET_CONFIGS = {
    "banking200k": {
        "local_parquet": str(BASE_DIR / "data" / "banking_intent_200k.parquet"),
        "text_col": "text",
        "label_col": "label_text",
        "domain": "enterprise-banking-110-intents",
        "total_rows": 220000,
        "num_intents": 110,
    },
    "massive": {
        "hf_name": "SetFit/amazon_massive_intent_en-US",
        "text_col": "text",
        "label_col": "label_text",
        "domain": "multi-domain-voice-assistant",
    },
    "banking77": {
        "hf_name": "PolyAI/banking77",
        "text_col": "text",
        "label_col": "label",
        "domain": "banking-and-finance",
    },
}

# ---------------------------------------------------------
# 3. Model Backbones
# ---------------------------------------------------------
# Intent Classifier Backbone
CLASSIFIER_BACKBONE = "distilbert/distilbert-base-uncased"
LEARNING_RATE = 5e-5
NUM_EPOCHS = 5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
LABEL_SMOOTHING_FACTOR = 0.05  # Improves probability calibration & accuracy

# Local Large Language Model (Qwen 4B loaded into GPU)
# Target model requested: Qwen 4B.
# We configure "Qwen/Qwen1.5-4B-Chat" as default, with auto-fallback to pre-cached
# local snapshot (e.g. Qwen2.5-1.5B or custom path) if offline.
LOCAL_LLM_MODEL_ID = "Qwen/Qwen1.5-4B-Chat"
FALLBACK_LOCAL_CACHE = os.path.expanduser(
    r"~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
)

# ---------------------------------------------------------
# 4. Algorithm Configurations
# ---------------------------------------------------------
# Algorithm 1: Conformal Prediction & Ambiguity-Aware Router
CONFORMAL_ALPHA = 0.10  # 90% guaranteed coverage prediction sets (1 - alpha)
OOD_ENTROPY_THRESHOLD = 0.85  # Out-Of-Domain anomaly threshold

# Algorithm 2: Multi-Objective Contextual Recommender (MMR + Exploration)
MMR_LAMBDA = 0.65       # Trade-off between intent relevance (0.65) and diversity (0.35)
UTILITY_WEIGHT = 0.20   # Weight for direct actionability score
EXPLORATION_BONUS = 0.10 # UCB exploration bonus for cold/emerging recommendations
TOP_K_RECOMMENDATIONS = 3

# Algorithm 3: Grounded Factuality & Compliance Guardrail
GROUNDEDNESS_THRESHOLD = 0.70 # Minimum semantic groundedness score
ENFORCE_PCI_DSS = True        # Scrub card numbers, CVVs, PINs
ENFORCE_ACTION_VERIFICATION = True # Ensure required safety steps are present

# ---------------------------------------------------------
# 5. OpenRouter Cloud LLM Fallback (Nemotron-3.5)
# ---------------------------------------------------------
# When local GPU is unavailable (Streamlit Cloud, CPU instance, or load failure),
# the engine seamlessly falls back to OpenRouter API with reasoning enabled.
# The API key is loaded strictly from secrets.toml (or Streamlit secrets/env) to avoid hardcoding.
def _load_secret(key: str, default: str = "") -> str:
    # 1. Direct secrets.toml files (.streamlit/secrets.toml or root secrets.toml)
    for p in [BASE_DIR / ".streamlit" / "secrets.toml", BASE_DIR / "secrets.toml"]:
        if p.exists():
            try:
                import tomllib
                with open(p, "rb") as f:
                    toml_data = tomllib.load(f)
                    if key in toml_data and toml_data[key]:
                        return str(toml_data[key]).strip()
            except Exception:
                pass

    # 2. Streamlit secrets context if running inside Streamlit
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key]).strip()
    except Exception:
        pass

    # 3. Environment variable
    env_val = os.environ.get(key)
    if env_val:
        return env_val.strip()

    return default


OPENROUTER_API_KEY = _load_secret("OPENROUTER_API_KEY", default="")
OPENROUTER_MODEL = _load_secret("OPENROUTER_MODEL", default="nvidia/nemotron-3.5-super-vl:free")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_ENABLE_REASONING = True



