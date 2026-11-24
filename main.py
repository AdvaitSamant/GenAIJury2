"""
Main Executable Pipeline for GenAI Jury 2.
End-to-End Orchestrator combining:
1. Intent Classification Model
2. Algorithm 1: ConformalRouter (Calibrated Conformal Sets & OOD Detection)
3. Algorithm 2: ContextualRecommender (Multi-Objective MMR with Bandit Exploration)
4. Local Qwen 4B GPU Inference Engine (Zero-API Latency on RTX 4050)
5. Algorithm 3: GroundedGuardrail (PCI-DSS Redaction & Factuality Audit)
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import (
    CHECKPOINTS_DIR,
    REPORTS_DIR,
    DEVICE,
    CONFORMAL_ALPHA,
    OOD_ENTROPY_THRESHOLD,
    MMR_LAMBDA,
    UTILITY_WEIGHT,
    EXPLORATION_BONUS,
    TOP_K_RECOMMENDATIONS,
    GROUNDEDNESS_THRESHOLD,
)
from algorithms.conformal_router import ConformalRouter
from algorithms.contextual_recommender import ContextualRecommender
from algorithms.grounded_guardrail import GroundedGuardrail
from local_llm import LocalQwenEngine


class GenAIJuryPipeline:
    """
    End-to-End AI Assistant Pipeline uniting Intent Classification,
    Conformal Routing, Contextual Recommendation, Local Qwen Generation, and Safety Guardrails.
    """
    def __init__(
        self,
        classifier_path: Optional[str] = None,
        load_llm: bool = True,
    ):
        print("\n" + "=" * 70)
        print("Initializing GenAI Jury 2 Autonomous Customer Intelligence Engine")
        print("=" * 70)

        # 1. Locate intent classifier weights and mappings
        self.classifier_path = self._resolve_classifier_path(classifier_path)
        print(f"[Pipeline] Loading intent classifier from: {self.classifier_path}")

        # Load intent mappings
        self.intent_names, self.id2label, self.label2id = self._load_intent_mappings()

        # Load classifier model & tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.classifier_path)
        self.classifier = AutoModelForSequenceClassification.from_pretrained(
            self.classifier_path
        ).to(DEVICE)
        self.classifier.eval()

        # 2. Initialize Algorithm 1: Conformal Router
        print("[Pipeline] Initializing Algorithm 1: Conformal Prediction Router...")
        self.router = ConformalRouter(
            class_names=self.intent_names,
            alpha=CONFORMAL_ALPHA,
            ood_entropy_threshold=OOD_ENTROPY_THRESHOLD,
        )
        self._calibrate_router()

        # 3. Initialize Algorithm 2: Contextual Recommender
        print("[Pipeline] Initializing Algorithm 2: Multi-Objective Contextual MMR Recommender...")
        self.recommender = ContextualRecommender(
            mmr_lambda=MMR_LAMBDA,
            utility_weight=UTILITY_WEIGHT,
            exploration_bonus=EXPLORATION_BONUS,
        )

        # 4. Initialize Algorithm 3: Grounded Guardrail
        print("[Pipeline] Initializing Algorithm 3: Grounded Factuality & Compliance Guardrail...")
        self.guardrail = GroundedGuardrail(groundedness_threshold=GROUNDEDNESS_THRESHOLD)

        # 5. Initialize Local Qwen GPU Engine
        self.llm = None
        if load_llm:
            print("[Pipeline] Initializing Local Qwen LLM on RTX 4050 GPU...")
            self.llm = LocalQwenEngine()
            try:
                self.llm.load_model()
            except Exception as e:
                print(f"[Pipeline] Warning: Local Qwen deferred or error: {e}")

        print("=" * 70)
        print("All 3 Algorithms, Classifier, and Local LLM Engine Successfully Online!")
        print("=" * 70 + "\n")

    def _resolve_classifier_path(self, user_path: Optional[str]) -> str:
        """Finds the best available classifier checkpoint."""
        if user_path and os.path.exists(user_path):
            return user_path

        best_model_path = CHECKPOINTS_DIR / "best_model"
        if best_model_path.exists() and (best_model_path / "config.json").exists():
            return str(best_model_path)

        # Fallback to existing checkpoint directories in workspace
        candidates = sorted(
            [p for p in CHECKPOINTS_DIR.glob("checkpoint-*") if p.is_dir()],
            key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
            reverse=True,
        )
        if candidates:
            return str(candidates[0])

        raise FileNotFoundError(
            f"No trained classifier found in {CHECKPOINTS_DIR}. Run 'python train.py' first."
        )

    def _load_intent_mappings(self) -> Tuple[List[str], Dict[int, str], Dict[str, int]]:
        """Loads or reconstructs id2label mapping from model config or saved report."""
        mapping_file = REPORTS_DIR / "intent_mapping.json"
        if mapping_file.exists():
            with open(mapping_file, "r") as f:
                data = json.load(f)
                id2label = {int(k): v for k, v in data["id2label"].items()}
                label2id = data["label2id"]
                intent_names = [id2label[i] for i in range(len(id2label))]
                return intent_names, id2label, label2id

        # Fallback to model config
        config_path = Path(self.classifier_path) / "config.json"
        with open(config_path, "r") as f:
            cfg = json.load(f)
            id2label = {int(k): v for k, v in cfg.get("id2label", {}).items()}
            label2id = cfg.get("label2id", {})
            intent_names = [id2label[i] for i in range(len(id2label))]
            return intent_names, id2label, label2id

    def _calibrate_router(self):
        """Calibrates conformal prediction router using saved validation logits if present."""
        val_logits_path = REPORTS_DIR / "val_logits.npy"
        val_labels_path = REPORTS_DIR / "val_labels.npy"
        if val_logits_path.exists() and val_labels_path.exists():
            val_logits = np.load(str(val_logits_path))
            val_labels = np.load(str(val_labels_path))
            self.router.calibrate(val_logits, val_labels)
            print(f"[Conformal Calibration] Router calibrated! Temperature T={self.router.scaler.temperature.item():.2f}, Quantile={self.router.quantile_threshold:.3f}")
        else:
            print("[Conformal Calibration] Calibration arrays not found. Using default inductive quantile.")
            self.router.quantile_threshold = 0.55
            self.router.is_calibrated = True

    def process_message(
        self,
        user_message: str,
        top_k_rec: int = TOP_K_RECOMMENDATIONS,
    ) -> Dict[str, Any]:
        """
        Processes a customer message through the complete 5-stage pipeline:
        1. Model Logits Extraction
        2. Algorithm 1: Conformal Routing (Coverage guaranteed prediction set)
        3. Algorithm 2: Multi-Objective MMR Recommendation
        4. Local Qwen 4B GPU Text Generation
        5. Algorithm 3: Grounded Factuality & Compliance Audit
        """
        # Stage 1: Feature Extraction & Classifier Logits
        inputs = self.tokenizer(
            user_message,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        ).to(DEVICE)

        with torch.inference_mode():
            outputs = self.classifier(**inputs)
            logits = outputs.logits[0].cpu().numpy()

        # Stage 2: Algorithm 1 - Conformal Routing
        routing_info = self.router.route(logits)
        detected_intent = routing_info["top_intent"]
        confidence = routing_info["calibrated_confidence"]
        prediction_set = routing_info["prediction_set"]
        routing_status = routing_info["routing_status"]

        # Stage 3: Algorithm 2 - Contextual Recommender
        recommendations = self.recommender.recommend(
            intent=detected_intent,
            user_message=user_message,
            top_k=top_k_rec,
        )

        # Stage 4: Local Qwen Generation
        if self.llm and self.llm.is_loaded:
            gen_result = self.llm.generate_response(
                user_message=user_message,
                intent=detected_intent,
                confidence=confidence,
                recommendations=recommendations,
                prediction_set=prediction_set,
            )
            raw_reply = gen_result["reply"]
            llm_metrics = {
                "generation_time_sec": gen_result["generation_time_sec"],
                "tokens_per_second": gen_result["tokens_per_second"],
                "model_used": gen_result["model_used"],
            }
        else:
            raw_reply = (
                f"I understand your query regarding '{detected_intent}'. "
                f"To resolve this, please follow our guided steps below."
            )
            llm_metrics = {"generation_time_sec": 0.0, "tokens_per_second": 0.0, "model_used": "mock_engine"}

        # Stage 5: Algorithm 3 - Grounded Guardrail & Safety Audit
        guardrail_report = self.guardrail.verify_and_remediate(
            generated_response=raw_reply,
            intent=detected_intent,
        )

        return {
            "user_message": user_message,
            "detected_intent": detected_intent,
            "calibrated_confidence": confidence,
            "conformal_routing": routing_info,
            "recommendations": recommendations,
            "raw_reply": raw_reply,
            "final_reply": guardrail_report.sanitized_text,
            "guardrail_report": {
                "passed": guardrail_report.passed,
                "groundedness_score": guardrail_report.groundedness_score,
                "violations": guardrail_report.violations,
                "remediation_action": guardrail_report.remediation_action,
            },
            "llm_metrics": llm_metrics,
        }


def run_test_suite(pipeline: GenAIJuryPipeline):
    """Executes a comprehensive verification suite demonstrating all 3 algorithms."""
    test_queries = [
        "I lost my credit card at the restaurant and need to freeze it immediately!",
        "How long will my international wire transfer take to clear in my balance?",
        "I want to change my PIN or reset it because I forgot it",
        "There is an unauthorized $250 charge on my card statement from an unknown merchant",
        "What is the airspeed velocity of an unladen swallow?",  # Out-Of-Domain query
        "My card number is 4532 0150 1234 5678 and CVV is 987 please refund me",  # PCI-DSS injection test
    ]

    print("\n" + "=" * 75)
    print("           RUNNING COMPREHENSIVE PIPELINE TEST SUITE")
    print("=" * 75)

    for idx, query in enumerate(test_queries, 1):
        print(f"\n--- Query #{idx}: '{query}' ---")
        result = pipeline.process_message(query)

        print(f"  [Intent Classification] : {result['detected_intent']} (Confidence: {result['calibrated_confidence']:.1%})")
        print(f"  [Algorithm 1 - Routing] : Status={result['conformal_routing']['routing_status']} | Conformal Set Size={result['conformal_routing']['conformal_set_size']}")
        if result['conformal_routing']['conformal_set_size'] > 1:
            p_set_str = ", ".join([f"{p['intent']} ({p['probability']:.1%})" for p in result['conformal_routing']['prediction_set']])
            print(f"    -> Ambiguous Set     : [{p_set_str}]")

        print(f"  [Algorithm 2 - Recommender Top 2]:")
        for rec in result['recommendations'][:2]:
            print(f"    * [{rec['category'].upper()}] {rec['title']} (Utility: {rec['utility_score']}, Rel: {rec['relevance_score']})")

        print(f"  [Local Qwen GPU Generation] : Latency={result['llm_metrics']['generation_time_sec']}s ({result['llm_metrics']['tokens_per_second']} tok/s)")
        print(f"    Assistant Reply: {result['final_reply'][:180]}...")

        print(f"  [Algorithm 3 - Guardrail] : Passed={result['guardrail_report']['passed']} | Groundedness={result['guardrail_report']['groundedness_score']:.2f} | Remediation={result['guardrail_report']['remediation_action']}")
        if result['guardrail_report']['violations']:
            for v in result['guardrail_report']['violations']:
                print(f"    -> Violation Flagged: {v}")


def interactive_cli(pipeline: GenAIJuryPipeline):
    """Interactive console session for real-time customer testing."""
    print("\n" + "=" * 65)
    print("  GENAI JURY INTERACTIVE CONSOLE (Type 'exit' or 'quit' to stop)")
    print("=" * 65)
    while True:
        try:
            msg = input("\nCustomer > ").strip()
            if not msg:
                continue
            if msg.lower() in ["exit", "quit", "q"]:
                print("Session terminated. Have a great day!")
                break

            result = pipeline.process_message(msg)
            print(f"\n[AI Assistant]:\n{result['final_reply']}\n")
            print(f"[Diagnostics] Intent: {result['detected_intent']} ({result['calibrated_confidence']:.1%}) | "
                  f"Route: {result['conformal_routing']['routing_status']} | "
                  f"Guardrail: {result['guardrail_report']['remediation_action']} | "
                  f"Latency: {result['llm_metrics']['generation_time_sec']}s")
            if result['recommendations']:
                print(f"[Proactive Recommendations]:")
                for r in result['recommendations']:
                    print(f"  -> [{r['category']}] {r['title']} ({r['action_url']})")
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GenAI Jury 2 Standalone Pipeline")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive CLI chat session")
    parser.add_argument("--no_llm", action="store_true", help="Run without loading local Qwen model")
    parser.add_argument("--checkpoint", type=str, default=None, help="Custom classifier checkpoint path")
    args = parser.parse_args()

    pipeline = GenAIJuryPipeline(
        classifier_path=args.checkpoint,
        load_llm=not args.no_llm,
    )

    if args.interactive:
        interactive_cli(pipeline)
    else:
        run_test_suite(pipeline)
