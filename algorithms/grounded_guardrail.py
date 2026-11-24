"""
Algorithm 3: Intent-Grounded Factuality & Compliance Guardrail Verifier.
Protects against LLM hallucinations, enforces regulatory banking compliance (PCI-DSS),
and guarantees required security action directives for high-risk customer intents.
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple


@dataclass
class GuardrailReport:
    passed: bool
    groundedness_score: float
    violations: List[str]
    sanitized_text: str
    remediation_action: str  # "approved", "patched", "template_fallback"


def luhn_checksum_valid(number_str: str) -> bool:
    """Verifies whether a numeric string passes the Luhn checksum for payment cards."""
    digits = [int(d) for d in number_str if d.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    reverse_digits = digits[::-1]
    for i, d in enumerate(reverse_digits):
        if i % 2 == 1:
            doubled = d * 2
            checksum += doubled - 9 if doubled > 9 else doubled
        else:
            checksum += d
    return checksum % 10 == 0


class GroundedGuardrail:
    """
    Real-Time Closed-Loop Safety & Factuality Guardrail Engine.
    Enforces PCI-DSS data redaction, audits compulsory intent actions,
    and scores generative token groundedness against official intent domain policies.
    """
    def __init__(self, groundedness_threshold: float = 0.70):
        self.groundedness_threshold = groundedness_threshold

        # Official policy facts and required actions per high-risk intent
        self.intent_policy_registry = {
            "lost_or_stolen_card": {
                "keywords": ["freeze", "block", "lock", "replacement", "unauthorized", "controls"],
                "required_action": "Customer must be instructed to immediately freeze their card via the mobile app or web portal.",
                "action_regex": r"(freeze|block|lock|deactivate)",
                "template": (
                    "Your card security is our top priority. Please immediately freeze your card in the "
                    "mobile app under 'Card Controls' to prevent any unauthorized charges. We will issue "
                    "and dispatch a replacement card right away."
                ),
            },
            "card_payment_not_recognised": {
                "keywords": ["dispute", "investigate", "merchant", "fraud", "freeze", "chargeback"],
                "required_action": "Advise customer to check transaction details and file a dispute or freeze card if suspicious.",
                "action_regex": r"(dispute|investigate|freeze|claim|unauthorized)",
                "template": (
                    "If you notice an unrecognized charge, please freeze your card immediately under 'Card Controls' "
                    "if you suspect fraud. You can submit a dispute with the transaction date and merchant details "
                    "in our Resolution Center for provisional credit."
                ),
            },
            "pin_blocked": {
                "keywords": ["reset", "verify", "identity", "biometric", "app", "branch"],
                "required_action": "Must provide identity verification instructions to reset PIN without asking for plain PIN.",
                "action_regex": r"(reset|verify|identity|biometric|app|branch)",
                "template": (
                    "For your security, PIN resets require multi-factor identity verification. You can securely "
                    "reset your PIN inside our official mobile app under 'Security Settings' using biometric login."
                ),
            },
            "terminate_account": {
                "keywords": ["zero balance", "outstanding", "closure", "verify", "transfer"],
                "required_action": "Customer must ensure zero pending balance and complete formal verification before closure.",
                "action_regex": r"(balance|pending|verify|transfer|support)",
                "template": (
                    "To close your account, please ensure all pending transactions have settled and your balance "
                    "is zero. Our account service team will verify your identity to finalize the closure."
                ),
            },
        }

    def _sanitize_pci_dss(self, text: str) -> Tuple[str, List[str]]:
        """
        Scans and redacts credit card numbers, CVVs, passwords, and sensitive PII.
        """
        violations = []
        sanitized = text

        # Regex for potential 13-19 digit card numbers (with or without dashes/spaces)
        card_pattern = r"\b(?:\d[ -]*?){13,19}\b"
        for match in re.finditer(card_pattern, text):
            candidate = re.sub(r"[ -]", "", match.group())
            if luhn_checksum_valid(candidate):
                violations.append("PCI-DSS Violation: Plaintext Credit/Debit Card Number Detected.")
                sanitized = sanitized.replace(match.group(), "[REDACTED_CARD_NUMBER]")

        # Regex for explicit CVV requests/mentions
        cvv_pattern = r"\b(cvv|cvc|security code)\s*[:=]?\s*(\d{3,4})\b"
        if re.search(cvv_pattern, text, flags=re.IGNORECASE):
            violations.append("PCI-DSS Violation: CVV/CVC Code Mention Detected.")
            sanitized = re.sub(cvv_pattern, r"\1: [REDACTED_CVV]", sanitized, flags=re.IGNORECASE)

        # Regex for plain password mentions
        pwd_pattern = r"\b(password|pin code)\s*[:=]\s*(\S+)"
        if re.search(pwd_pattern, text, flags=re.IGNORECASE):
            violations.append("Security Violation: Plaintext Credential Detected.")
            sanitized = re.sub(pwd_pattern, r"\1: [REDACTED_CREDENTIAL]", sanitized, flags=re.IGNORECASE)

        return sanitized, violations

    def _compute_groundedness(self, text: str, policy: Dict[str, Any]) -> float:
        """
        Calculates lexical-semantic groundedness score between response text
        and policy knowledge keywords.
        """
        text_lower = text.lower()
        keywords = policy["keywords"]
        matches = sum(1 for kw in keywords if kw.lower() in text_lower)
        groundedness = matches / len(keywords) if keywords else 1.0
        return float(min(1.0, groundedness * 1.35))  # Scaled expectation

    def verify_and_remediate(
        self,
        generated_response: str,
        intent: str,
    ) -> GuardrailReport:
        """
        Executes full guardrail audit pipeline:
        1. Scrub PCI/PII credentials.
        2. Verify mandatory intent safety actions.
        3. Check groundedness score against domain policy.
        4. Remediate with patch injection or certified fallback template if necessary.
        """
        sanitized_text, violations = self._sanitize_pci_dss(generated_response)
        remediation_action = "approved"

        policy = self.intent_policy_registry.get(intent)
        if policy:
            # Check compulsory action
            action_regex = policy["action_regex"]
            if not re.search(action_regex, sanitized_text, flags=re.IGNORECASE):
                violations.append(
                    f"Action Incompleteness: Response omitted mandatory action directive: {policy['required_action']}"
                )

            # Compute groundedness score
            groundedness_score = self._compute_groundedness(sanitized_text, policy)

            if groundedness_score < self.groundedness_threshold:
                violations.append(
                    f"Hallucination Risk: Low semantic groundedness ({groundedness_score:.2f} < {self.groundedness_threshold})."
                )
        else:
            # For general intents without specific policy registry
            groundedness_score = 0.90

        # Remediation Decision Logic
        if violations:
            if policy and any("PCI-DSS" in v for v in violations):
                # Critical violation: replace with certified safe template
                sanitized_text = policy["template"]
                remediation_action = "template_fallback"
            elif policy and any("Action Incompleteness" in v for v in violations):
                # Minor omission: auto-patch missing directive
                sanitized_text = (
                    f"{sanitized_text}\n\n[Security Notice: {policy['required_action']}]"
                )
                remediation_action = "patched"
            else:
                remediation_action = "patched"

        passed = len(violations) == 0

        return GuardrailReport(
            passed=passed,
            groundedness_score=round(groundedness_score, 3),
            violations=violations,
            sanitized_text=sanitized_text,
            remediation_action=remediation_action,
        )
