from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal


# ── Seed generation ──────────────────────────────────────────────────────────

class SeedKeyword(BaseModel):
    keyword: str

    @field_validator("keyword")
    @classmethod
    def validate_keyword(cls, v: str) -> str:
        v = v.strip().lower()
        words = v.split()
        if not (2 <= len(words) <= 4):
            raise ValueError(
                f"keyword '{v}' must be 2-4 words, got {len(words)}. "
                "Reject single words and full sentences."
            )
        return v


class SeedKeywordList(BaseModel):
    keywords: list[SeedKeyword]
    intent_stage: str   # model echoes back; used for drift detection
    rationale: str      # one-sentence reasoning — logged for diagnostics


# ── Keyword pool records ──────────────────────────────────────────────────────

class BaseKeyword(BaseModel):
    """Minimal keyword record — pre-expansion."""
    keyword: str
    source_stage: str                           # TOFU | MOFU | BOFU


class ExpandedKeyword(BaseKeyword):
    """Post Keyword Planner expansion — adds volume and bid data."""
    avg_monthly_searches: Optional[int] = None
    competition: str = "UNKNOWN"
    competition_index: Optional[float] = None
    low_top_of_page_bid_micros: Optional[int] = None
    high_top_of_page_bid_micros: Optional[int] = None


class PooledKeyword(ExpandedKeyword):
    """Post-clustering — adds cluster assignment and intent label."""
    cluster_id: Optional[int] = None
    assigned_intent: Optional[str] = None      # TOFU | MOFU | BOFU | MIXED
    intent_confidence: Optional[float] = None
    dominant_modifier: Optional[str] = None    # geo | price | comparison |
                                               # feature | emergency | brand | None
    was_mixed: bool = False                    # True if MIXED before adjudication
    ad_group_name: Optional[str] = None


# ── Cluster and pool structures ───────────────────────────────────────────────

class ClusterRecord(BaseModel):
    """One cluster / ad-group candidate. Built by cluster_inference node."""
    cluster_id: int
    assigned_intent: str
    keyword_count: int
    total_monthly_volume: int
    avg_monthly_volume: float
    avg_competition_index: Optional[float]
    representative_keyword: str         # highest-volume keyword in cluster
    top_keywords: list[str]             # top 5 by volume
    dominant_modifier_type: Optional[str]
    ad_group_name: str                  # title-cased rep keyword — deterministic


class FunnelPool(BaseModel):
    """One funnel stage's keyword pool, post-clustering."""
    stage: str                          # TOFU | MOFU | BOFU
    keywords: list[str]
    cluster_ids: list[int]
    total_volume: int


class KeywordPoolMap(BaseModel):
    """
    Primary output of the Keyword Agent consumed by Strategy Agent
    and Bidding Agent. Does not contain raw DataFrames — only
    serialisable data.
    """
    business_key: str
    business_name: str
    total_keywords: int
    pools: dict[str, list[str]]         # stage → list of keyword strings
    clusters: list[ClusterRecord]
    silhouette_score: Optional[float]
    metadata: dict = Field(default_factory=dict)


# ── Cluster inference ─────────────────────────────────────────────────────────

class GapCluster(BaseModel):
    cluster_id: int
    ad_group_name: str
    assigned_intent: str
    avg_monthly_volume: float
    competition_index: Optional[float]
    gap_score: float                    # avg_volume / (competition_index + 1)


class MissingStageFlag(BaseModel):
    stage: str
    keyword_count: int
    diagnosis: Literal["seeding_failure", "market_characteristic", "unknown"]
    note: str


class ClusterInferenceReport(BaseModel):
    """
    Output of the deterministic Cluster Inference Node.
    Consumed by Strategy Agent to ground targeting decisions in
    actual keyword market data rather than prior assumptions.
    """
    business_key: str
    intent_distribution: dict[str, float]       # stage → volume share, sums to 1.0
    dominant_stage: str
    dominant_stage_volume_share: float
    gap_clusters: list[GapCluster]
    competitive_pressure: dict[str, Optional[float]]  # stage → mean competition_index
    missing_stages: list[MissingStageFlag]
    recommended_focus: Literal["single_stage", "bofu_gap", "flank", "balanced"]
    recommended_focus_reasoning: str             # must cite specific numbers


# ── LLM adjudication ─────────────────────────────────────────────────────────

class IntentAdjudication(BaseModel):
    """Structured output for MIXED cluster LLM adjudication."""
    cluster_id: int
    assigned_intent: Literal["TOFU", "MOFU", "BOFU"]
    confidence: Literal["HIGH", "MEDIUM", "LOW"]
    reasoning: str


# ── AdGroupContext — output boundary of the Keyword Agent ────────────────────

class AdGroupContext(BaseModel):
    """
    Structured brief passed to the Copy Agent. This is the final output
    of the Keyword Agent. No raw keyword lists cross this boundary —
    only pre-interpreted, structured briefs.
    """
    # Identity
    ad_group_name: str
    campaign_name: str
    intent_stage: str                   # preserves MIXED for transparency
    cluster_id: int

    # Keyword signals (derived from cluster — not the raw list)
    representative_keyword: str
    top_keywords: list[str]             # top 5 by volume — actual user language
    avg_monthly_volume: float
    total_monthly_volume: int
    competition_level: str              # "LOW" | "MEDIUM" | "HIGH" | "UNKNOWN"
    dominant_modifier_type: Optional[str]

    # Theme summary — deterministic, no LLM
    topic: str                          # cleaned title-cased rep keyword
    user_goal: str                      # intent taxonomy template + topic injected
    pain_point: str                     # from MODIFIER_PAIN_POINT lookup

    # Business context (passed through from upstream)
    business_name: str
    relevant_service: str
    key_differentiator: str
    landing_page_url: str

    # Copy constraints (consumed by Copy Agent)
    headline_char_limit: int = 30
    description_char_limit: int = 90
    max_headlines: int = 15
    max_descriptions: int = 4
    must_include_keyword: bool = True
