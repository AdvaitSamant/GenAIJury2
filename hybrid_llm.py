"""
Hybrid LLM Manager for GenAI Jury 2.
Coordinates primary Local GPU inference (Qwen 4B on NVIDIA RTX 4050)
with seamless automated cloud fallback to OpenRouter API ('nvidia/nemotron-3.5-super-vl:free').
Ensures the deployed Streamlit app always possesses generative LLM intelligence
even on CPU / Streamlit Cloud environments.
"""

import os
from typing import Any, Dict, List, Optional
import torch

from config import GPU_NAME, TOTAL_VRAM_GB
from local_llm import LocalQwenEngine
from openrouter_llm import OpenRouterEngine


class HybridLLMManager:
    """
    Manages the multi-tier LLM inference cascade:
    1. Primary Tier: Local Qwen 4B / 1.5B running in CUDA VRAM (RTX 4050).
    2. Fallback Tier: OpenRouter API (Nemotron-3.5-Super-VL with reasoning).
    3. Safety Net Tier: Deterministic Certified Banking Policy Engine.
    """

    def __init__(
        self,
        prefer_local: bool = True,
        mode: str = "auto",
        openrouter_api_key: Optional[str] = None,
    ):
        self.prefer_local = prefer_local
        self.mode = mode  # 'auto', 'api_only', 'local_only'

        # Initialize engines
        self.local_engine = LocalQwenEngine()
        self.api_engine = OpenRouterEngine(api_key=openrouter_api_key)

        self.local_loaded = False
        self.local_error: Optional[str] = None
        self.api_error: Optional[str] = None

    def initialize(self):
        """Attempts to initialize the primary local model if GPU and mode permit."""
        if self.mode == "api_only":
            print("[HybridLLM] Mode set to 'api_only'. OpenRouter Nemotron-3.5 is primary.")
            return

        if not torch.cuda.is_available():
            self.local_error = "CUDA unavailable (Cloud / CPU environment detected)."
            print(f"[HybridLLM] {self.local_error} OpenRouter API will serve as active LLM engine.")
            return

        try:
            print("[HybridLLM] Attempting to load primary Local Qwen into GPU VRAM...")
            self.local_engine.load_model()
            self.local_loaded = self.local_engine.is_loaded
            if self.local_loaded:
                print(f"[HybridLLM] Primary Local GPU model online ({self.local_engine.loaded_model_id}).")
        except Exception as exc:
            self.local_loaded = False
            self.local_error = str(exc)
            print(f"[HybridLLM] Primary Local Qwen failed to load: {exc}")
            print("[HybridLLM] OpenRouter API fallback is primed and ready.")

    def set_mode(self, mode: str):
        """Switches runtime execution mode: 'auto', 'api_only', or 'local_only'."""
        if mode in ("auto", "api_only", "local_only"):
            self.mode = mode
            if mode != "api_only" and not self.local_loaded and torch.cuda.is_available():
                try:
                    self.local_engine.load_model()
                    self.local_loaded = self.local_engine.is_loaded
                except Exception as exc:
                    self.local_error = str(exc)

    def generate_response(
        self,
        user_message: str,
        intent: str,
        confidence: float,
        recommendations: Optional[List[Dict[str, Any]]] = None,
        prediction_set: Optional[List[Dict[str, Any]]] = None,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        policy_fallback_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generates customer assistant response cascading through Local -> API -> Policy.
        """
        # Tier 1: Local GPU Inference (if mode allows and model is loaded)
        if self.mode in ("auto", "local_only") and self.local_loaded:
            try:
                local_res = self.local_engine.generate_response(
                    user_message=user_message,
                    intent=intent,
                    confidence=confidence,
                    recommendations=recommendations,
                    prediction_set=prediction_set,
                )
                return {
                    "reply": local_res["reply"],
                    "engine_type": "local",
                    "engine_name": "Local GPU (Qwen)",
                    "model_used": local_res.get("model_used", "Qwen-Local"),
                    "generation_time_sec": local_res["generation_time_sec"],
                    "tokens_per_second": local_res["tokens_per_second"],
                    "tokens_generated": local_res.get("tokens_generated", 0),
                    "is_fallback": False,
                    "fallback_reason": None,
                    "reasoning_details": None,
                }
            except Exception as exc:
                print(f"[HybridLLM] Local GPU generation failed during runtime: {exc}")
                self.local_error = f"Runtime generation failed: {exc}"
                if self.mode == "local_only":
                    raise

        # Tier 2: OpenRouter API Fallback (if mode is 'api_only' or auto fallback triggered)
        if self.mode != "local_only" and self.api_engine.is_available():
            fallback_reason = None
            if self.mode == "auto":
                if not torch.cuda.is_available():
                    fallback_reason = "Local GPU unavailable (Cloud/CPU environment). OpenRouter API active."
                elif not self.local_loaded:
                    fallback_reason = f"Local model load failed ({self.local_error or 'offline'}). OpenRouter API active."
                elif self.local_error:
                    fallback_reason = f"Local generation failed at runtime: {self.local_error}."

            try:
                api_res = self.api_engine.generate_response(
                    user_message=user_message,
                    intent=intent,
                    confidence=confidence,
                    recommendations=recommendations,
                    prediction_set=prediction_set,
                    chat_history=chat_history,
                )
                return {
                    "reply": api_res["reply"],
                    "engine_type": "api",
                    "engine_name": "OpenRouter API (Nemotron 3.5)",
                    "model_used": api_res.get("model_used", self.api_engine.model),
                    "generation_time_sec": api_res["generation_time_sec"],
                    "tokens_per_second": api_res["tokens_per_second"],
                    "tokens_generated": api_res.get("tokens_generated", 0),
                    "is_fallback": (self.mode == "auto"),
                    "fallback_reason": fallback_reason,
                    "reasoning_details": api_res.get("reasoning_details"),
                }
            except Exception as exc:
                print(f"[HybridLLM] OpenRouter API fallback error: {exc}")
                self.api_error = str(exc)

        # Tier 3: Deterministic Policy Engine (Safety Net)
        reply = policy_fallback_text or (
            f"I understand your query regarding **{intent.replace('_', ' ').title()}**. "
            f"Please follow the certified institutional guidance and actions provided below."
        )
        if self.api_error:
            fallback_msg = f"Cloud API error: {self.api_error}. Falling back to certified policy engine."
        elif not self.api_engine.is_available():
            fallback_msg = "OpenRouter API key is missing. Please configure it in Streamlit Secrets or sidebar."
        else:
            fallback_msg = "Both Local GPU and OpenRouter API unavailable. Certified policy engine invoked."

        return {
            "reply": reply,
            "engine_type": "policy",
            "engine_name": "Certified Policy Engine",
            "model_used": "cloud_policy_rules",
            "generation_time_sec": 0.02,
            "tokens_per_second": 0.0,
            "tokens_generated": len(reply.split()),
            "is_fallback": True,
            "fallback_reason": fallback_msg,
            "reasoning_details": None,
        }

    def get_status(self) -> Dict[str, Any]:
        """Provides status metadata for the Streamlit UI."""
        allocated_vram = (
            torch.cuda.memory_allocated() / (1024 ** 3)
            if torch.cuda.is_available()
            else 0.0
        )
        active_type = "local"
        if self.mode == "api_only":
            active_type = "api"
        elif not self.local_loaded:
            active_type = "api" if self.api_engine.is_available() else "policy"

        return {
            "mode": self.mode,
            "active_type": active_type,
            "local_loaded": self.local_loaded,
            "local_model": getattr(self.local_engine, "loaded_model_id", "Qwen 4B"),
            "local_error": self.local_error,
            "vram_gb": allocated_vram,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": GPU_NAME,
            "api_available": self.api_engine.is_available(),
            "api_model": self.api_engine.model,
            "api_error": self.api_error,
        }
