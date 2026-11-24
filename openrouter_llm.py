"""
OpenRouter Cloud LLM Engine for Streamlit Cloud Fallback & Zero-GPU Deployment.
Uses 'nvidia/nemotron-3.5-super-vl:free' via OpenRouter API with chain-of-thought reasoning enabled.
Preserves reasoning_details across dialogue turns.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import requests

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_API_URL,
    OPENROUTER_ENABLE_REASONING,
    OPENROUTER_MODEL,
)


class OpenRouterEngine:
    """
    Manages cloud LLM generation through the OpenRouter API.
    Used as an immediate fallback when local GPU weights fail to load
    or in headless / cloud environments (e.g., Streamlit Cloud).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = OPENROUTER_MODEL,
        api_url: str = OPENROUTER_API_URL,
        enable_reasoning: bool = OPENROUTER_ENABLE_REASONING,
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY") or OPENROUTER_API_KEY
        self.model = model
        self.api_url = api_url
        self.enable_reasoning = enable_reasoning

    def is_available(self) -> bool:
        """Checks if a valid API key is present."""
        return bool(self.api_key and len(self.api_key.strip()) > 10)

    def test_connection(self) -> Tuple[bool, str]:
        """Performs a lightweight test to verify API key validity and network access."""
        if not self.is_available():
            return False, "OpenRouter API key is missing or empty."

        headers = {
            "Authorization": f"Bearer {self.api_key.strip()}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": "Respond with 'OK'."}],
            "max_tokens": 10,
            "reasoning": {"enabled": False},
        }
        try:
            resp = requests.post(self.api_url, headers=headers, json=data, timeout=25)
            if resp.status_code == 200:
                return True, "OpenRouter API connected successfully."
            return False, f"OpenRouter returned HTTP {resp.status_code}: {resp.text[:120]}"
        except Exception as exc:
            return False, f"Connection failed: {str(exc)}"

    def generate_response(
        self,
        user_message: str,
        intent: str,
        confidence: float,
        recommendations: Optional[List[Dict[str, Any]]] = None,
        prediction_set: Optional[List[Dict[str, Any]]] = None,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.6,
        timeout: int = 45,
    ) -> Dict[str, Any]:
        """
        Generates an intent-grounded, empathetic response via OpenRouter Nemotron-3.5.
        Preserves reasoning_details across conversation turns if provided in chat_history.
        """
        if not self.is_available():
            raise ValueError(
                "OpenRouter API key is not configured. Provide OPENROUTER_API_KEY in environment or config."
            )

        # Build context-aware system prompt matching local Qwen instructions
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

        # Assemble messages payload with reasoning_details preserved
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]

        # Append previous conversation history if available
        if chat_history:
            for turn in chat_history:
                role = turn.get("role")
                content = turn.get("content", "")
                if role == "user":
                    messages.append({"role": "user", "content": content})
                elif role == "assistant":
                    asst_msg: Dict[str, Any] = {"role": "assistant", "content": content}
                    # Preserve reasoning_details unmodified if present
                    if "reasoning_details" in turn and turn["reasoning_details"]:
                        asst_msg["reasoning_details"] = turn["reasoning_details"]
                    messages.append(asst_msg)

        # Append current user query
        messages.append({"role": "user", "content": user_message})

        headers = {
            "Authorization": f"Bearer {self.api_key.strip()}",
            "Content-Type": "application/json",
        }

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if self.enable_reasoning:
            payload["reasoning"] = {"enabled": True}

        start_time = time.perf_counter()
        resp = requests.post(
            self.api_url,
            headers=headers,
            data=json.dumps(payload),
            timeout=timeout,
        )
        generation_time = time.perf_counter() - start_time

        if resp.status_code != 200:
            raise RuntimeError(
                f"OpenRouter API request failed [HTTP {resp.status_code}]: {resp.text}"
            )

        resp_json = resp.json()
        choices = resp_json.get("choices", [])
        if not choices:
            raise RuntimeError(f"OpenRouter returned empty choices: {resp_json}")

        message_obj = choices[0].get("message", {})
        raw_content = message_obj.get("content")
        reply_content = (raw_content or "").strip()
        reasoning_details = message_obj.get("reasoning_details")

        # If content was empty due to token limit or raw thinking output, fallback gracefully
        if not reply_content:
            if message_obj.get("reasoning"):
                reply_content = (
                    f"I have reviewed your query regarding **{intent.replace('_', ' ').title()}**. "
                    f"Please proceed with the certified action steps below to safely resolve your request."
                )
            else:
                reply_content = (
                    f"I understand your query regarding **{intent.replace('_', ' ').title()}**. "
                    f"Please follow the institutional guidance provided."
                )

        # Extract token counts
        usage = resp_json.get("usage", {})
        completion_tokens = usage.get("completion_tokens")
        if not completion_tokens:
            completion_tokens = max(1, len(reply_content) // 4)

        tokens_per_sec = (
            completion_tokens / generation_time if generation_time > 0 else 0.0
        )


        return {
            "reply": reply_content,
            "tokens_generated": completion_tokens,
            "generation_time_sec": round(generation_time, 3),
            "tokens_per_second": round(tokens_per_sec, 1),
            "model_used": self.model,
            "provider": "openrouter_api",
            "engine_type": "api",
            "device": "Cloud API (OpenRouter)",
            "reasoning_details": reasoning_details,
        }
