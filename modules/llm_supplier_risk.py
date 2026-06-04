import json
import os
from openai import OpenAI
from pydantic import BaseModel, Field, model_validator
from modules.prompts import SUPPLIER_RISK_SYSTEM_PROMPT
from modules.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class SupplierRiskAssessment(BaseModel):
    risk_tier: str = Field(description="Overall risk level: LOW | MEDIUM | HIGH | CRITICAL")
    compounding_factors: list[str] = Field(description="Risks that *multiply* each other's impact.")
    strategic_action: str = Field(
        description="Action: DUAL_SOURCE | EMERGENCY_STOCK | TERMINATE | MONITOR | EXPEDITE | ACCEPT_DELAY"
    )
    rationale: str = Field(
        description="Causal chain: WHY do these specific compounding factors force THIS specific action?"
    )
    confidence_score: int = Field(description="0–100. How confident is the assessment?")
    what_would_change_assessment: str = Field(
        description="One concrete piece of information that would alter the strategy."
    )

    @model_validator(mode="before")
    @classmethod
    def _normalise_keys(cls, data: dict) -> dict:
        """
        Grok sometimes returns synonymous field names.
        This maps them to canonical names before Pydantic validates.
        """
        aliases = {
            "reasoning":        "rationale",
            "explanation":      "rationale",
            "analysis":         "rationale",
            "justification":    "rationale",
            "causal_chain":     "rationale",
            "what_would_change":        "what_would_change_assessment",
            "pivot_condition":          "what_would_change_assessment",
            "change_condition":         "what_would_change_assessment",
            "what_changes_assessment":  "what_would_change_assessment",
            "compounding_risks": "compounding_factors",
            "risk_factors":      "compounding_factors",
            "factors":           "compounding_factors",
            "action":             "strategic_action",
            "recommended_action": "strategic_action",
            "strategy":           "strategic_action",
            "confidence": "confidence_score",
            "score":      "confidence_score",
        }
        return {aliases.get(k, k): v for k, v in data.items()}


_GROK_MODEL = "grok-4-1-fast-reasoning"   

# Injected at the end of the system prompt so Grok knows the exact key names
_JSON_SCHEMA_INSTRUCTION = """
You MUST respond with a JSON object using EXACTLY these keys (no others, no renaming):
{
  "risk_tier": "CRITICAL | HIGH | MEDIUM | LOW",
  "compounding_factors": ["string", "string"],
  "strategic_action": "DUAL_SOURCE | EMERGENCY_STOCK | TERMINATE | MONITOR | EXPEDITE | ACCEPT_DELAY",
  "rationale": "string — causal chain explaining why these factors force this action",
  "confidence_score": 0-100,
  "what_would_change_assessment": "string — one specific data point that would change the strategy"
}
No preamble. No markdown. No extra keys. Valid JSON only.
"""

def _build_demo_response(text: str) -> SupplierRiskAssessment:
    """
    Keyword-based fallback used when XAI_API_KEY is missing or the API call fails.
    Every field is derived from the actual input — nothing is hardcoded.
    """
    text_lower = text.lower()

    # Risk signal detection
    critical_keywords = [
        "sole source", "sole-source", "only supplier", "strike",
        "bankruptcy", "insolvent", "shutdown", "sole supplier",
    ]
    high_keywords = [
        "flood", "fire", "financial distress", "no alternative",
        "single source", "plant closure", "force majeure",
    ]
    medium_keywords = [
        "maintenance", "partial", "shortage", "backlog",
        "delay", "disruption", "constrained",
    ]
    low_keywords = [
        "minor", "alternative available", "secondary line",
        "buffer", "sufficient stock", "no impact",
    ]

    critical_hits = [w for w in critical_keywords if w in text_lower]
    high_hits     = [w for w in high_keywords     if w in text_lower]
    medium_hits   = [w for w in medium_keywords   if w in text_lower]
    low_hits      = [w for w in low_keywords      if w in text_lower]

    if len(critical_hits) >= 2 or (critical_hits and high_hits):
        risk_tier        = "CRITICAL"
        strategic_action = "DUAL_SOURCE"
        confidence_score = 40
    elif critical_hits or len(high_hits) >= 2:
        risk_tier        = "HIGH"
        strategic_action = "EMERGENCY_STOCK"
        confidence_score = 35
    elif high_hits or len(medium_hits) >= 2:
        risk_tier        = "MEDIUM"
        strategic_action = "EXPEDITE"
        confidence_score = 30
    elif medium_hits or not low_hits:
        risk_tier        = "LOW"
        strategic_action = "MONITOR"
        confidence_score = 25
    else:
        risk_tier        = "LOW"
        strategic_action = "ACCEPT_DELAY"
        confidence_score = 25

    # Compounding factors: built from what was actually detected 
    factors = []
    if critical_hits:
        factors.append(
            f"Critical signals in the update: {', '.join(f'\"{w}\"' for w in critical_hits)}"
        )
    if high_hits:
        factors.append(
            f"High-risk signals detected: {', '.join(f'\"{w}\"' for w in high_hits)}"
        )
    if medium_hits and not (critical_hits or high_hits):
        factors.append(
            f"Moderate disruption signals: {', '.join(f'\"{w}\"' for w in medium_hits)}"
        )
    if not factors:
        factors = [
            "No strong compounding risk signals detected — review manually for context-specific risks"
        ]

    excerpt = text.strip().replace("\n", " ")
    if len(excerpt) > 250:
        excerpt = excerpt[:250] + "..."

    if risk_tier == "CRITICAL":
        pivot = (
            "If a pre-qualified alternative supplier exists and can cover demand within the disruption window, "
            "DUAL_SOURCE urgency drops and EMERGENCY_STOCK becomes the preferred short-term action."
        )
    elif risk_tier == "HIGH":
        pivot = (
            "If current on-hand inventory at the buyer's DC covers demand for longer than the estimated disruption period, "
            "EMERGENCY_STOCK becomes unnecessary and the action shifts to MONITOR."
        )
    elif risk_tier == "MEDIUM":
        pivot = (
            "If the supplier confirms a firm recovery date within 7 days and stock buffers are adequate, "
            "action can be downgraded from EXPEDITE to ACCEPT_DELAY."
        )
    else:
        pivot = (
            "If new information reveals sole-source dependency or financial distress at the supplier, "
            "risk tier would escalate immediately to HIGH or CRITICAL."
        )

    return SupplierRiskAssessment(
        risk_tier=risk_tier,
        compounding_factors=factors,
        strategic_action=strategic_action,
        rationale=(
            f"[Demo mode — XAI_API_KEY not set. This is a keyword-based summary, not a full AI assessment.]\n\n"
            f"Input received: \"{excerpt}\"\n\n"
            f"Keyword analysis flagged a {risk_tier} tier based on the signals above, "
            f"driving the {strategic_action} recommendation. "
            f"Confidence is intentionally low ({confidence_score}/100) — set XAI_API_KEY in your .env "
            f"for a full Grok-powered causal analysis."
        ),
        confidence_score=confidence_score,
        what_would_change_assessment=pivot,
    )


def analyze_supplier_risk(text: str) -> SupplierRiskAssessment:
    logger.info("Initiating supplier risk assessment.")
    api_key = os.getenv("XAI_API_KEY", "")

    if not api_key:
        logger.warning("XAI_API_KEY not found. Falling back to keyword-based demo response.")
        return _build_demo_response(text)

    try:
        logger.debug(f"Configuring xAI Grok client with model: {_GROK_MODEL}")
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
        )

        response = client.chat.completions.create(
            model=_GROK_MODEL,
            temperature=0.2,
            max_tokens=1024,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": SUPPLIER_RISK_SYSTEM_PROMPT + "\n\n" + _JSON_SCHEMA_INSTRUCTION,
                },
                {
                    "role": "user",
                    "content": f"Supplier update:\n\n{text}",
                },
            ],
        )

        raw = response.choices[0].message.content.strip()
        logger.debug(f"Raw Grok response (first 300 chars): {raw[:300]}")

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        parsed = json.loads(raw)
        logger.debug(f"Parsed keys from Grok: {list(parsed.keys())}")

        assessment = SupplierRiskAssessment(**parsed)
        logger.info(
            f"Risk assessment complete. Tier: {assessment.risk_tier}, "
            f"Action: {assessment.strategic_action}, "
            f"Confidence: {assessment.confidence_score}"
        )
        return assessment

    except Exception as exc:
        logger.error(f"Grok API call failed: {exc}", exc_info=True)
        result = _build_demo_response(text)
        result.rationale = (
            f"[API error — falling back to keyword analysis]\n\nError: {exc}\n\n"
            + result.rationale
        )
        return result