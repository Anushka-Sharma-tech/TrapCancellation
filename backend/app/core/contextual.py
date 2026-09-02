import re
from typing import Dict, Any, List

class ContextualRiskEngine:
    """
    Analyzes conversation transcripts and metadata to compute conversational threat risk.
    Detects urgency escalation, banking fraud phrases, OTP harvesting, and authority coercion.
    """

    HIGH_RISK_PATTERNS = [
        r"\b(transfer|neft|rtgs|imps|wire|send money|funds?)\b",
        r"\b(urgent|immediately|right now|emergency|asap|critical)\b",
        r"\b(otp|one time password|pin|cvv|password|passcode)\b",
        r"\b(vendor payment|tax clearance|customs penalty|account freeze)\b",
        r"\b(don't call|keep this confidential|whatsapp only|don't discuss with team)\b",
    ]

    # Indic & Hinglish contextual threat markers
    HINGLISH_PATTERNS = [
        r"\b(jaldi karo|turant|bhejo|paisa|khata|rupaye|paise|account mein)\b",
        r"\b(kisi ko mat batana|secret hai|call cut mat karna)\b",
    ]

    def __init__(self):
        self.compiled_rules = [
            re.compile(pattern, re.IGNORECASE) for pattern in (self.HIGH_RISK_PATTERNS + self.HINGLISH_PATTERNS)
        ]

    def analyze_intent(self, transcript: str) -> Dict[str, Any]:
        """
        Scans call transcript for social engineering vectors.
        Returns context risk score (0-100) and triggered indicators.
        """
        if not transcript or not transcript.strip():
            return {"context_score": 0, "triggers": [], "urgency_flag": False}

        matched_triggers: List[str] = []
        score = 0

        for pattern in self.compiled_rules:
            matches = pattern.findall(transcript)
            if matches:
                matched_triggers.extend([str(m) for m in matches])
                score += 20

        # Heuristic modifiers
        has_urgency = bool(re.search(r"(urgent|immediately|turant|emergency|jaldi)", transcript, re.I))
        has_financial = bool(re.search(r"(transfer|money|paisa|funds|rupaye|neft|imps)", transcript, re.I))

        # Compounding rule: Urgency + Financial transaction request
        if has_urgency and has_financial:
            score += 30

        final_score = min(100, score)

        return {
            "context_score": final_score,
            "triggers": list(set(matched_triggers)),
            "urgency_flag": has_urgency,
            "coercion_detected": final_score >= 60
        }

    def evaluate_metadata(self, metadata: Dict[str, Any]) -> int:
        """
        Evaluates session metadata (call origin, transaction value, hour of call).
        """
        risk_adjustment = 0
        amount = metadata.get("transaction_amount_inr", 0)
        
        # Transactions over 2 Lakh INR get elevated scrutiny
        if amount >= 200000:
            risk_adjustment += 25
        elif amount >= 50000:
            risk_adjustment += 10

        if metadata.get("caller_number_spoofed_hint", False):
            risk_adjustment += 40

        return risk_adjustment