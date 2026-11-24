"""
Algorithms Package for GenAI Jury 2
Contains 3 major high-impact algorithms:
1. ConformalRouter: Temperature-Scaled Conformal Prediction with Ambiguity & OOD Routing.
2. ContextualRecommender: Multi-Objective Maximal Marginal Relevance (MMR) with Bandit Exploration.
3. GroundedGuardrail: Intent-Grounded Factuality Verifier and Regulatory Safety Guardrail.
"""

from .conformal_router import ConformalRouter, TemperatureScaler
from .contextual_recommender import ContextualRecommender, RecommendationItem
from .grounded_guardrail import GroundedGuardrail, GuardrailReport

__all__ = [
    "ConformalRouter",
    "TemperatureScaler",
    "ContextualRecommender",
    "RecommendationItem",
    "GroundedGuardrail",
    "GuardrailReport",
]
