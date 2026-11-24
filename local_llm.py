"""
Local Qwen LLM Engine for High-Speed GPU Inference.
Loads Qwen model directly onto the NVIDIA GeForce RTX 4050 Laptop GPU,
replacing external API dependencies with zero-latency local generation.
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import (
    LOCAL_LLM_MODEL_ID,
    FALLBACK_LOCAL_CACHE,
    DEVICE,
    TOTAL_VRAM_GB,
)


class LocalQwenEngine:
    """
    Manages local Qwen inference loaded in CUDA memory.
    Optimized for RTX 4050 Laptop GPU (6GB VRAM) with FP16 precision.
    """
    def __init__(
        self,
        model_name_or_path: str = None,
        fallback_path: str = FALLBACK_LOCAL_CACHE,
        device: str = DEVICE,
        prefer_local_cache: bool = True,
    ):
        self.target_model = model_name_or_path or LOCAL_LLM_MODEL_ID
        self.fallback_path = fallback_path
        self.device = device
        self.prefer_local_cache = prefer_local_cache
        self.tokenizer = None
        self.model = None
        self.loaded_model_id = None
        self.is_loaded = False

    def load_model(self):
        """
        Loads the Qwen model into GPU VRAM with float16 precision.
        Checks locally cached weights first for instant zero-download execution,
        then falls back to remote download if requested.
        """
        if self.is_loaded:
            return

        if not torch.cuda.is_available():
            print("[LocalQwenEngine] CUDA unavailable (CPU/Cloud instance detected). Using certified policy response generator.")
            self.is_loaded = False
            return

        candidates = []
        if self.prefer_local_cache and self.fallback_path and os.path.exists(self.fallback_path):
            candidates.append((self.fallback_path, True))
            candidates.append((self.target_model, False))
        else:
            candidates.append((self.target_model, False))
            if self.fallback_path and os.path.exists(self.fallback_path):
                candidates.append((self.fallback_path, True))

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device_map = "auto" if torch.cuda.is_available() else None

        for candidate, is_local in candidates:
            print(f"[LocalQwenEngine] Attempting to load Qwen from: {candidate} on {self.device}...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    candidate,
                    trust_remote_code=True,
                    padding_side="left",
                    local_files_only=is_local,
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                self.model = AutoModelForCausalLM.from_pretrained(
                    candidate,
                    dtype=dtype,
                    device_map=device_map,
                    trust_remote_code=True,
                    local_files_only=is_local,
                )
                self.model.eval()
                self.loaded_model_id = candidate
                self.is_loaded = True

                allocated_vram = (
                    torch.cuda.memory_allocated() / (1024 ** 3)
                    if torch.cuda.is_available()
                    else 0.0
                )
                print(
                    f"[LocalQwenEngine] Successfully loaded '{Path(candidate).name if is_local else candidate}' onto GPU. "
                    f"Allocated VRAM: {allocated_vram:.2f} GB / {TOTAL_VRAM_GB:.1f} GB."
                )
                return
            except Exception as exc:
                print(f"[LocalQwenEngine] Candidate '{candidate}' not loaded: {exc}")

        raise RuntimeError(
            f"Could not load any Qwen model candidate on {self.device}. "
            f"Ensure HuggingFace weights or local snapshots exist."
        )

    def generate_response(
        self,
        user_message: str,
        intent: str,
        confidence: float,
        recommendations: Optional[List[Dict[str, Any]]] = None,
        prediction_set: Optional[List[Dict[str, Any]]] = None,
        max_new_tokens: int = 150,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        """
        Generates an intent-grounded, proactive response using local GPU Qwen.
        """
        if not self.is_loaded:
            self.load_model()

        # Build context-aware system prompt
        recommendation_bullets = ""
        if recommendations:
            rec_titles = [f"- {r['title']}: {r['description']}" for r in recommendations[:2]]
            recommendation_bullets = (
                "\nRecommended proactive solutions for the user:\n"
                + "\n".join(rec_titles)
            )

        conformal_info = ""
        if prediction_set and len(prediction_set) > 1:
            ambiguous_intents = ", ".join([p["intent"] for p in prediction_set])
            conformal_info = (
                f"\nNote: The user query was flagged as ambiguous between possible intents: [{ambiguous_intents}]. "
                f"Kindly address the primary intent '{intent}', but offer quick clarification if needed."
            )

        system_prompt = (
            f"You are an expert, empathetic, and highly secure AI banking and customer support assistant. "
            f"The customer's message has been verified with intent: '{intent}' "
            f"(calibrated confidence: {confidence:.0%})."
            f"{conformal_info}"
            f"{recommendation_bullets}\n\n"
            f"Instructions:\n"
            f"1. Directly answer the customer's request concisely, accurately, and professionally.\n"
            f"2. Incorporate any necessary security directives (e.g. freezing card if lost, verifying identity if PIN blocked).\n"
            f"3. Never ask for full credit card numbers, CVV codes, or passwords.\n"
            f"4. Proactively guide them to the next best action."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_message}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        start_time = time.perf_counter()
        with torch.inference_mode():
            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generation_time = time.perf_counter() - start_time
        new_tokens = output_tokens[0][input_len:]
        response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        tokens_per_sec = len(new_tokens) / generation_time if generation_time > 0 else 0.0

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "reply": response_text,
            "tokens_generated": len(new_tokens),
            "generation_time_sec": round(generation_time, 3),
            "tokens_per_second": round(tokens_per_sec, 1),
            "model_used": self.loaded_model_id,
            "device": str(self.model.device),
        }
