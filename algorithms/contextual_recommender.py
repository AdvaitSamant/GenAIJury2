"""
Algorithm 2: Multi-Objective Contextual Maximal Marginal Relevance (MMR) Recommender
with Dynamic Bandit Exploration.
Bridges intent classification with proactive action recommendations, tutorials, and self-service flows,
avoiding repetitive filter bubbles and balancing relevance, diversity, and actionable utility.
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RecommendationItem:
    id: str
    title: str
    category: str  # "action", "guide", "video_tutorial", "safety"
    description: str
    action_url: str
    intent_tags: List[str]
    utility_score: float  # 0.0 to 1.0 (higher for immediate self-service resolution)
    impressions: int = 0
    clicks: int = 0


# Default rich catalog of financial, banking, and service recommendations
DEFAULT_CATALOG = [
    RecommendationItem(
        id="act_freeze_card",
        title="Emergency 1-Click Card Freeze",
        category="action",
        description="Instantly toggle card freeze in your mobile app to prevent unauthorized charges.",
        action_url="app://security/card-freeze",
        intent_tags=["lost_or_stolen_card", "card_payment_not_recognised", "card_swallowed"],
        utility_score=0.95,
    ),
    RecommendationItem(
        id="act_order_replacement",
        title="Order Instant Replacement Card",
        category="action",
        description="Ship a replacement contactless card with express delivery to your registered address.",
        action_url="app://cards/order-replacement",
        intent_tags=["lost_or_stolen_card", "card_arrival", "damaged_card"],
        utility_score=0.90,
    ),
    RecommendationItem(
        id="act_dispute_transaction",
        title="Initiate Chargeback & Fraud Dispute",
        category="action",
        description="Submit unrecognized charge details for provisional credit and merchant chargeback.",
        action_url="app://disputes/file-claim",
        intent_tags=["card_payment_not_recognised", "wrong_amount_of_cash_received", "extra_charge"],
        utility_score=0.92,
    ),
    RecommendationItem(
        id="act_reset_pin",
        title="Secure PIN Reset & Unblock Portal",
        category="action",
        description="Verify your identity with biometric authentication and set a new 4-digit PIN instantly.",
        action_url="app://security/reset-pin",
        intent_tags=["pin_blocked", "change_pin", "forgot_pin"],
        utility_score=0.88,
    ),
    RecommendationItem(
        id="act_virtual_card",
        title="Generate Disposable Virtual Card",
        category="action",
        description="Create single-use virtual card details for safe online purchases while waiting for plastic card.",
        action_url="app://cards/virtual-cards",
        intent_tags=["get_disposable_virtual_card", "getting_virtual_card", "lost_or_stolen_card"],
        utility_score=0.85,
    ),
    RecommendationItem(
        id="guide_wire_timelines",
        title="Official Bank Transfer Clearing Timelines Guide",
        category="guide",
        description="Explains domestic SEPA/ACH vs SWIFT international clearing windows, cut-off hours, and holidays.",
        action_url="https://help.bank.com/transfers/clearing-windows",
        intent_tags=["balance_not_updated_after_bank_transfer", "transfer_timing", "pending_transfer"],
        utility_score=0.75,
    ),
    RecommendationItem(
        id="guide_identity_kyc",
        title="Why Identity Verification is Required by Regulation",
        category="guide",
        description="Detailed regulatory overview of AML/KYC compliance and acceptable photo IDs.",
        action_url="https://help.bank.com/compliance/kyc-guide",
        intent_tags=["why_verify_identity", "verify_my_identity", "identity_verification_pending"],
        utility_score=0.70,
    ),
    RecommendationItem(
        id="video_fraud_prevention",
        title="Video Tutorial: Spotting Phishing & Unrecognized Charges",
        category="video_tutorial",
        description="A 2-minute visual walkthrough on recognizing merchant billing descriptors and social engineering.",
        action_url="https://youtube.com/watch?v=bank_fraud_prevention",
        intent_tags=["card_payment_not_recognised", "card_security", "social_engineering"],
        utility_score=0.78,
    ),
    RecommendationItem(
        id="video_atm_troubleshooting",
        title="Video Tutorial: What to Do When an ATM Retains Cash or Card",
        category="video_tutorial",
        description="Visual guide on noting ATM terminal ID, receipt generation, and filing an immediate discrepancy claim.",
        action_url="https://youtube.com/watch?v=atm_retention_guide",
        intent_tags=["card_swallowed", "declined_cash_withdrawal", "wrong_amount_of_cash_received"],
        utility_score=0.82,
    ),
    RecommendationItem(
        id="guide_account_closure",
        title="Account Closure Checklist & Balance Transfer Procedure",
        category="guide",
        description="Requirements for zero-balance reconciliation, standing order transfers, and final tax statements.",
        action_url="https://help.bank.com/accounts/closing-checklist",
        intent_tags=["terminate_account", "close_account", "cancel_subscription"],
        utility_score=0.72,
    ),
]


class ContextualRecommender:
    """
    Multi-Objective MMR Recommender with UCB Bandit Exploration.
    Calculates item candidate embeddings, balances relevance and diversity,
    and returns top ranked proactive actions.
    """
    def __init__(
        self,
        catalog: Optional[List[RecommendationItem]] = None,
        mmr_lambda: float = 0.65,
        utility_weight: float = 0.20,
        exploration_bonus: float = 0.10,
    ):
        self.catalog = catalog or DEFAULT_CATALOG
        self.mmr_lambda = mmr_lambda
        self.utility_weight = utility_weight
        self.exploration_bonus = exploration_bonus
        self.total_recommendation_requests = 10

        # Build TF-IDF vectorizer over catalog textual representations
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        corpus = [
            f"{item.title} {item.description} {' '.join(item.intent_tags)} {item.category}"
            for item in self.catalog
        ]
        self.item_matrix = self.vectorizer.fit_transform(corpus)

    def _compute_ucb_bonus(self, item: RecommendationItem) -> float:
        """Computes Upper Confidence Bound exploration bonus."""
        n_item = max(1, item.impressions)
        bonus = math.sqrt(2.0 * math.log(max(2, self.total_recommendation_requests)) / n_item)
        return float(min(1.0, bonus * 0.1))

    def recommend(
        self,
        intent: str,
        user_message: str = "",
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Runs Multi-Objective MMR Ranking:
        MMR(d) = lambda * Sim(d, q) - (1-lambda) * max_{s in S} Sim(d, s) + beta * Utility(d) + gamma * UCB(d)
        """
        self.total_recommendation_requests += 1

        # Query representation combining diagnosed intent and original user utterance
        query_text = f"{intent.replace('_', ' ')} {user_message}"
        query_vec = self.vectorizer.transform([query_text])

        # Base relevance similarities: shape (1, N)
        relevance_sims = cosine_similarity(query_vec, self.item_matrix)[0]

        # Tag-based exact match boost
        for idx, item in enumerate(self.catalog):
            if intent in item.intent_tags:
                relevance_sims[idx] += 0.35  # Intent match prior

        # Inter-item pairwise similarities for diversity calculation: shape (N, N)
        item_sims = cosine_similarity(self.item_matrix, self.item_matrix)

        selected_indices: List[int] = []
        candidate_indices = list(range(len(self.catalog)))

        while len(selected_indices) < min(top_k, len(self.catalog)):
            best_idx = -1
            best_score = -float("inf")

            for idx in candidate_indices:
                rel = relevance_sims[idx]
                item = self.catalog[idx]

                # Max similarity to already selected items
                if selected_indices:
                    max_sim_to_selected = max(item_sims[idx, s_idx] for s_idx in selected_indices)
                else:
                    max_sim_to_selected = 0.0

                # Multi-Objective MMR Score formula
                utility = item.utility_score * self.utility_weight
                exploration = self._compute_ucb_bonus(item) * self.exploration_bonus
                mmr_core = self.mmr_lambda * rel - (1.0 - self.mmr_lambda) * max_sim_to_selected

                total_score = mmr_core + utility + exploration

                if total_score > best_score:
                    best_score = total_score
                    best_idx = idx

            if best_idx != -1:
                selected_indices.append(best_idx)
                candidate_indices.remove(best_idx)
                # Increment impressions for online bandit statistics
                self.catalog[best_idx].impressions += 1
            else:
                break

        results = []
        for rank, idx in enumerate(selected_indices, start=1):
            item = self.catalog[idx]
            results.append({
                "rank": rank,
                "id": item.id,
                "title": item.title,
                "category": item.category,
                "description": item.description,
                "action_url": item.action_url,
                "relevance_score": round(float(relevance_sims[idx]), 3),
                "utility_score": round(item.utility_score, 2),
                "impressions": item.impressions,
            })

        return results
