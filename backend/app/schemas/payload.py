from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class CallMetadata(BaseModel):
    caller_id: str = Field(..., example="+919876543210")
    target_account_id: Optional[str] = Field(None, example="ACC-774921")
    transaction_amount_inr: Optional[float] = Field(0.0, example=250000.0)
    caller_number_spoofed_hint: bool = Field(False)
    language_hint: Optional[str] = Field("en-IN", example="hi-IN")

class CallAnalysisRequest(BaseModel):
    call_id: str
    metadata: CallMetadata
    transcript: Optional[str] = None
    audio_base64: Optional[str] = None

class RiskBreakdown(BaseModel):
    acoustic_score: int
    prosody_score: int
    contextual_score: int
    metadata_penalty: int

class CallAnalysisResponse(BaseModel):
    call_id: str
    overall_risk_score: int
    threat_level: str  # NORMAL, ELEVATED, HIGH_RISK, CRITICAL_SPOOF
    is_spoofed: bool
    recommended_action: str
    breakdown: RiskBreakdown
    detected_triggers: List[str]