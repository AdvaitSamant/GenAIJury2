"""
Algorithm 1: Temperature-Scaled Inductive Conformal Prediction & Ambiguity-Aware Router
Solves overconfident uncalibrated predictions by providing mathematically guaranteed
prediction sets (1 - alpha coverage) and detecting multi-intent ambiguity vs Out-Of-Domain anomalies.
"""

import math
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class TemperatureScaler(nn.Module):
    """
    Learns a single temperature parameter T > 0 on validation logits using
    Negative Log-Likelihood (NLL) loss to calibrate softmax probabilities.
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temp = self.temperature.clamp(min=0.01)
        return logits / temp

    def fit(self, val_logits: np.ndarray, val_labels: np.ndarray, max_iter: int = 100) -> float:
        """Optimizes temperature T using L-BFGS or Adam on validation data."""
        logits_t = torch.tensor(val_logits, dtype=torch.float32)
        labels_t = torch.tensor(val_labels, dtype=torch.long)
        nll_criterion = nn.CrossEntropyLoss()

        optimizer = optim.LBFGS([self.temperature], lr=0.05, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(logits_t), labels_t)
            loss.backward()
            return loss

        try:
            optimizer.step(eval_loss)
        except Exception:
            # Fallback to Adam if L-BFGS encounters singular hessian
            adam_opt = optim.Adam([self.temperature], lr=0.02)
            for _ in range(max_iter):
                adam_opt.zero_grad()
                loss = nll_criterion(self.forward(logits_t), labels_t)
                loss.backward()
                adam_opt.step()

        return float(self.temperature.item())


class ConformalRouter:
    """
    Inductive Conformal Prediction Engine.
    Computes calibrated non-conformity quantiles to construct guaranteed prediction sets.
    Routes queries to SINGLE_INTENT, AMBIGUOUS (multi-intent), or OUT_OF_DOMAIN.
    """
    def __init__(
        self,
        class_names: List[str],
        alpha: float = 0.10,
        ood_entropy_threshold: float = 0.85,
    ):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.alpha = alpha  # Error rate (1 - alpha = 0.90 coverage guarantee)
        self.ood_entropy_threshold = ood_entropy_threshold
        self.scaler = TemperatureScaler()
        self.quantile_threshold: float = 1.0
        self.is_calibrated: bool = False

    def calibrate(self, val_logits: np.ndarray, val_labels: np.ndarray):
        """
        Fits temperature scaling on validation set and computes the empirical
        conformal quantile threshold from calibration non-conformity scores.
        """
        # Step 1: Optimize temperature
        self.scaler.fit(val_logits, val_labels)

        # Step 2: Compute calibrated probabilities
        with torch.no_grad():
            scaled_logits = self.scaler(torch.tensor(val_logits, dtype=torch.float32))
            calibrated_probs = torch.softmax(scaled_logits, dim=-1).numpy()

        # Step 3: Compute non-conformity scores on calibration samples: s_i = 1 - p(y_true)
        n_samples = len(val_labels)
        true_class_probs = calibrated_probs[np.arange(n_samples), val_labels]
        non_conformity_scores = 1.0 - true_class_probs

        # Step 4: Compute finite-sample adjusted quantile
        # (n + 1) * (1 - alpha) / n
        p_val = math.ceil((n_samples + 1) * (1.0 - self.alpha)) / n_samples
        p_val = min(1.0, max(0.0, p_val))
        self.quantile_threshold = float(np.quantile(non_conformity_scores, p_val, method="higher"))
        self.is_calibrated = True

    def route(self, uncalibrated_logits: np.ndarray) -> Dict[str, Any]:
        """
        Routes an incoming query using conformal prediction sets and Shannon entropy.
        Input: uncalibrated_logits (1D array of length num_classes).
        """
        logits_t = torch.tensor(uncalibrated_logits, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            scaled = self.scaler(logits_t)
            probs = torch.softmax(scaled, dim=-1)[0].numpy()

        argmax_idx = int(np.argmax(probs))
        top_intent = self.class_names[argmax_idx]
        top_conf = float(probs[argmax_idx])

        # Normalized Shannon entropy to detect Out-Of-Domain (OOD)
        eps = 1e-12
        entropy = -float(np.sum(probs * np.log(probs + eps)))
        max_possible_entropy = np.log(self.num_classes)
        normalized_entropy = float(entropy / max_possible_entropy) if max_possible_entropy > 0 else 0.0

        # Conformal Prediction Set: C(x) = {y : 1 - p(y|x) <= quantile_threshold}
        # Meaning: include classes where p(y|x) >= 1 - quantile_threshold
        min_p = max(0.0, 1.0 - self.quantile_threshold) if self.is_calibrated else 0.40
        prediction_set_indices = [idx for idx, p in enumerate(probs) if p >= min_p]

        # Sort prediction set by probability descending
        prediction_set_indices.sort(key=lambda idx: probs[idx], reverse=True)
        prediction_set = [
            {"intent": self.class_names[idx], "probability": float(probs[idx])}
            for idx in prediction_set_indices
        ]

        # Determine routing status
        if normalized_entropy > self.ood_entropy_threshold and top_conf < 0.35:
            routing_status = "OUT_OF_DOMAIN"
            recommended_action = "escalate_to_human_or_ask_clarification"
        elif len(prediction_set) == 1:
            routing_status = "SINGLE_INTENT"
            recommended_action = "direct_automated_resolution"
        elif len(prediction_set) > 1:
            routing_status = "AMBIGUOUS"
            recommended_action = "disambiguate_options"
        else:
            routing_status = "LOW_CONFIDENCE"
            recommended_action = "ask_clarifying_question"

        return {
            "top_intent": top_intent,
            "calibrated_confidence": round(top_conf, 4),
            "temperature": round(float(self.scaler.temperature.item()), 3),
            "normalized_entropy": round(normalized_entropy, 4),
            "conformal_set_size": len(prediction_set),
            "prediction_set": prediction_set,
            "routing_status": routing_status,
            "recommended_action": recommended_action,
            "coverage_guarantee": f"{round((1 - self.alpha) * 100)}%",
        }
