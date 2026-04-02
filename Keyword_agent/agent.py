"""
LangGraph graph definition for the Keyword Agent.

Usage:
    from keyword_agent import app, KeywordAgentState
    from keyword_agent.agent import init_clients

    llm, genai_client, googleads_client = init_clients(CONFIG)

    result = app.invoke(
        {
            "business_context":    my_business_ctx,
            "geo_targeting_config": {},
            "reseed_tracker":       {},
            "reseed_needed":        False,
            "errors": [], "warnings": [], "logs": [],
        },
        config={
            "configurable": {
                "llm":              llm,
                "genai_client":     genai_client,
                "googleads_client": googleads_client,
                "customer_id":      CUSTOMER_ID,
            }
        },
    )
"""

import os
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END

from .state import KeywordAgentState
from .config import CONFIG
from .nodes import (
    seed_generator,
    kw_planner_expansion,
    pool_inspection,
    adaptive_reseed,
    clustering,
    post_clustering_ops,
    mixed_adjudication,
    cluster_inference,
    adgroup_context_build,
)


# ─────────────────────────────────────────────────────────────────────────────
# Client initialisation
# ─────────────────────────────────────────────────────────────────────────────

def init_clients(config: dict):
    """
    Initialise all three clients once at startup.
    Inject into the graph via config["configurable"].

    Returns:
        (llm, genai_client, googleads_client)
    """
    # from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_groq import ChatGroq
    from google import genai
    from google.ads.googleads.client import GoogleAdsClient

    # llm = ChatGoogleGenerativeAI(
    #     model        = config["llm_model"],
    #     temperature  = config["llm_temperature"],
    #     google_api_key = os.environ["GEMINI_API_KEY"],
    # )

    llm = ChatGroq(
        model        = config["llm_model"],
        temperature  = config["llm_temperature"],
        api_key     = os.environ["GROQ_API_KEY"],
    )
    genai_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    googleads_client = GoogleAdsClient.load_from_storage(
        config["googleads_yaml_path"], version="v22"
    )
    return llm, genai_client, googleads_client


# ─────────────────────────────────────────────────────────────────────────────
# Routing functions
# ─────────────────────────────────────────────────────────────────────────────

def _stages_below_threshold(pool_coverage: dict) -> list[str]:
    """Return stage names that are below min_keywords_per_stage."""
    min_kw = CONFIG["inference"]["min_keywords_per_stage"]
    return [stage for stage, count in pool_coverage.items() if count < min_kw]


def route_after_inspection(state: KeywordAgentState) -> str:
    """
    Fire adaptive_reseed if any stage is below threshold AND the hard cap
    (reseed_tracker[stage] < 1) has not been reached yet.
    Otherwise proceed to clustering.
    """
    if not state.get("reseed_needed", False):
        return "clustering"

    pool_coverage  = state.get("pool_coverage", {})
    reseed_tracker = state.get("reseed_tracker", {})
    below          = _stages_below_threshold(pool_coverage)

    # Only route to reseed if at least one below-threshold stage hasn't hit the cap yet
    can_reseed = any(reseed_tracker.get(stage, 0) < 1 for stage in below)
    return "adaptive_reseed" if can_reseed else "clustering"


def route_after_reseed(state: KeywordAgentState) -> str:
    """
    After one reseed pass, always return to pool_inspection for a coverage re-check.
    pool_inspection will set reseed_needed=False if coverage is now sufficient,
    or leave it True — but route_after_inspection will then route to clustering
    anyway because reseed_tracker will show the hard cap is reached.
    """
    return "pool_inspection"


# ─────────────────────────────────────────────────────────────────────────────
# Graph definition
# ─────────────────────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    graph = StateGraph(KeywordAgentState)

    # Add nodes
    graph.add_node("seed_generator",       seed_generator)
    graph.add_node("kw_planner_expansion", kw_planner_expansion)
    graph.add_node("pool_inspection",      pool_inspection)
    graph.add_node("adaptive_reseed",      adaptive_reseed)
    graph.add_node("clustering",           clustering)
    graph.add_node("post_clustering_ops",  post_clustering_ops)
    graph.add_node("mixed_adjudication",   mixed_adjudication)
    graph.add_node("cluster_inference",    cluster_inference)
    graph.add_node("adgroup_context_build", adgroup_context_build)

    # Linear edges
    graph.add_edge(START,                  "seed_generator")
    graph.add_edge("seed_generator",       "kw_planner_expansion")
    graph.add_edge("kw_planner_expansion", "pool_inspection")

    # Conditional: re-seed or skip straight to clustering
    graph.add_conditional_edges(
        "pool_inspection",
        route_after_inspection,
        {"adaptive_reseed": "adaptive_reseed", "clustering": "clustering"},
    )
    graph.add_conditional_edges(
        "adaptive_reseed",
        route_after_reseed,
        {"pool_inspection": "pool_inspection"},
    )

    # Post-clustering linear chain
    graph.add_edge("clustering",           "post_clustering_ops")
    graph.add_edge("post_clustering_ops",  "mixed_adjudication")
    graph.add_edge("mixed_adjudication",   "cluster_inference")
    graph.add_edge("cluster_inference",    "adgroup_context_build")
    graph.add_edge("adgroup_context_build", END)

    return graph


# Compiled graph — import this to run the agent
app = _build_graph().compile()
