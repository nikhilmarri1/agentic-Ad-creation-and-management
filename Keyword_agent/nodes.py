"""
All 9 LangGraph node functions for the Keyword Agent.

Each node:
- Accepts (state: KeywordAgentState, config: RunnableConfig)
- Reads clients from config["configurable"]: llm, genai_client, googleads_client, customer_id
- Returns a partial state dict
- Catches exceptions and writes to state["errors"] or state["warnings"] rather than raising
"""

import os
import re
import time
import json
from typing import Optional

import numpy as np
import pandas as pd
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score as sk_silhouette_score
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist
import umap
import hdbscan

from .config import CONFIG, INTENT_TAXONOMY, STAGES, MODIFIER_PAIN_POINT, COMPETITION_LEVEL_MAP
from .state import KeywordAgentState
from .tools import embed_keywords, expand_keywords
from .schemas.keyword_agent import (
    SeedKeywordList,
    ClusterRecord,
    GapCluster,
    MissingStageFlag,
    ClusterInferenceReport,
    IntentAdjudication,
    AdGroupContext,
    KeywordPoolMap,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: extract clients from LangGraph configurable
# ─────────────────────────────────────────────────────────────────────────────

def _clients(config: RunnableConfig) -> tuple:
    """Return (llm, genai_client, googleads_client, customer_id) from configurable."""
    c = config.get("configurable", {})
    return (
        c["llm"],
        c["genai_client"],
        c["googleads_client"],
        c["customer_id"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Node 1: seed_generator
# ─────────────────────────────────────────────────────────────────────────────

def _build_seed_prompt(business_ctx: dict, stage: str, stage_def: dict) -> list:
    signal_words_str    = ", ".join(f'"{w}"' for w in stage_def["signal_words"])
    competitors_str     = ", ".join(business_ctx["competitors"])
    services_str        = ", ".join(business_ctx["services"])
    differentiators_str = "; ".join(business_ctx["differentiators"])

    geo_instruction = ""
    if business_ctx.get("is_local"):
        geo_instruction = (
            f"\n- IMPORTANT: Local business serving {business_ctx['geo']}. "
            f"At least 4 of your 8 seeds MUST include a geo-modifier "
            f"(e.g., 'Phoenix', 'Phoenix AZ', 'near me')."
        )

    system_msg = SystemMessage(content=(
        "You are a senior paid search strategist with 10+ years planning Google Ads campaigns "
        "across B2B SaaS, e-commerce, and local services. You understand search intent deeply "
        "and generate precise seed keywords that anchor Keyword Planner to a specific funnel stage."
    ))

    user_msg = HumanMessage(content=f"""Generate exactly {CONFIG['seeds_per_stage']} seed keywords for a Google Ads campaign.

Business: {business_ctx['name']}
Industry: {business_ctx['industry']}
Services/Products: {services_str}
Target Audience: {business_ctx['target_audience']}
Differentiators: {differentiators_str}
Competitors: {competitors_str}
Geography: {business_ctx['geo']}

FUNNEL STAGE: {stage} — {stage_def['name']}
User goal at this stage: {stage_def['user_goal']}
Signal words for this stage: {signal_words_str}
Campaign implication: {stage_def['campaign_implication']}

CONSTRAINTS (strictly enforce all):
- Each keyword must be 2 to 4 words only — reject single words and full sentences
- No brand names (yours or competitors')
- No generic filler: 'online', 'digital', 'software' unless directly relevant
- Keywords must match the {stage} intent stage exclusively
- Must be realistic queries a real human would type in Google{geo_instruction}

Return intent_stage as exactly: "{stage}"
""")
    return [system_msg, user_msg]


def _generate_seeds(
    business_ctx: dict,
    stage: str,
    llm,
    max_retries: int = 3,
) -> list[str]:
    stage_def  = INTENT_TAXONOMY[stage]
    messages   = _build_seed_prompt(business_ctx, stage, stage_def)
    structured = llm.with_structured_output(SeedKeywordList)

    for attempt in range(max_retries):
        try:
            result: SeedKeywordList = structured.invoke(messages)
            if result.intent_stage != stage:
                print(f"    WARNING: echo '{result.intent_stage}' != requested '{stage}'")
            print(f"    [{business_ctx['name']} / {stage}] {result.rationale}")
            return [sk.keyword for sk in result.keywords]
        except Exception as exc:
            wait = 2 ** attempt
            print(f"    Attempt {attempt+1} failed: {exc}. Retrying in {wait}s...")
            time.sleep(wait)

    raise RuntimeError(
        f"Seed generation failed for {business_ctx['name']}/{stage} after {max_retries} attempts"
    )


def seed_generator(state: KeywordAgentState, config: RunnableConfig) -> dict:
    """
    Node 1: Generate seed keywords for all 3 funnel stages via LLM.
    Writes: seeds_by_stage
    """
    llm, _, _, _ = _clients(config)
    business_ctx = state["business_context"]
    logs         = list(state.get("logs", []))
    errors       = list(state.get("errors", []))
    warnings     = list(state.get("warnings", []))

    print(f"\n[Node: seed_generator] {business_ctx['name']}")
    seeds_by_stage = {}

    for stage in STAGES:
        try:
            seeds = _generate_seeds(business_ctx, stage, llm)
            seeds_by_stage[stage] = seeds
            logs.append(f"seed_generator: {stage} → {seeds}")
            print(f"    {stage}: {seeds}")
        except Exception as exc:
            msg = f"seed_generator: failed for stage {stage}: {exc}"
            errors.append(msg)
            print(f"    ERROR: {msg}")
            # Return early — cannot proceed without seeds
            return {"errors": errors, "warnings": warnings, "logs": logs}

    # Persist seeds alongside outputs for reference
    outputs_dir = CONFIG["outputs_dir"]
    os.makedirs(outputs_dir, exist_ok=True)
    seeds_path = os.path.join(outputs_dir, "generated_seeds.json")
    existing   = {}
    if os.path.exists(seeds_path):
        try:
            with open(seeds_path) as f:
                existing = json.load(f)
        except Exception:
            pass
    existing[business_ctx["key"]] = seeds_by_stage
    with open(seeds_path, "w") as f:
        json.dump(existing, f, indent=2)

    return {
        "seeds_by_stage": seeds_by_stage,
        "errors":         errors,
        "warnings":       warnings,
        "logs":           logs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 2: kw_planner_expansion
# ─────────────────────────────────────────────────────────────────────────────

def kw_planner_expansion(state: KeywordAgentState, config: RunnableConfig) -> dict:
    """
    Node 2: Expand seeds via Google Ads Keyword Planner for all 3 stages.
    Writes: pool_df
    """
    _, _, googleads_client, customer_id = _clients(config)
    business_ctx  = state["business_context"]
    seeds_by_stage = state["seeds_by_stage"]
    logs    = list(state.get("logs", []))
    errors  = list(state.get("errors", []))
    warnings = list(state.get("warnings", []))

    print(f"\n[Node: kw_planner_expansion] {business_ctx['name']}")
    os.makedirs(CONFIG["outputs_dir"], exist_ok=True)

    stage_dfs = []
    for stage in STAGES:
        try:
            df = expand_keywords(
                business_ctx   = business_ctx,
                stage          = stage,
                seeds          = seeds_by_stage[stage],
                googleads_client = googleads_client,
                customer_id    = customer_id,
                config         = CONFIG,
            )
            stage_dfs.append(df)
            logs.append(f"kw_planner_expansion: {stage} → {len(df)} keywords")
            time.sleep(1.5)  # rate limit between API calls
        except Exception as exc:
            msg = f"kw_planner_expansion: failed for stage {stage}: {exc}"
            errors.append(msg)
            print(f"    ERROR: {msg}")

    if not stage_dfs:
        return {"errors": errors, "warnings": warnings, "logs": logs}

    # Concat TOFU → MOFU → BOFU (order matters for dedup priority)
    combined = pd.concat(stage_dfs, ignore_index=True)
    # Dedup: keep first occurrence (TOFU-first preserves stage priority)
    combined = combined.drop_duplicates(subset=["keyword"], keep="first").reset_index(drop=True)
    combined["avg_monthly_searches"] = combined["avg_monthly_searches"].fillna(0).astype(int)

    pool_path = os.path.join(CONFIG["outputs_dir"], f"{business_ctx['key']}_combined_pool.csv")
    combined.to_csv(pool_path, index=False)
    logs.append(f"kw_planner_expansion: combined pool → {len(combined)} unique keywords")
    print(f"  Combined pool: {len(combined)} unique keywords → {pool_path}")

    return {
        "pool_df":   combined,
        "errors":    errors,
        "warnings":  warnings,
        "logs":      logs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 3: pool_inspection
# ─────────────────────────────────────────────────────────────────────────────

def pool_inspection(state: KeywordAgentState, config: RunnableConfig) -> dict:
    """
    Node 3: Read-only coverage check. Sets reseed_needed flag.
    Writes: pool_coverage, reseed_needed
    """
    pool_df  = state["pool_df"]
    logs     = list(state.get("logs", []))
    errors   = list(state.get("errors", []))
    warnings = list(state.get("warnings", []))
    min_kw   = CONFIG["inference"]["min_keywords_per_stage"]

    print(f"\n[Node: pool_inspection]")
    print(f"  Total keywords: {len(pool_df)}")

    coverage = {}
    for stage in STAGES:
        count = int((pool_df["source_stage"] == stage).sum())
        coverage[stage] = count
        flag = "SPARSE" if count < min_kw else "OK"
        print(f"  {stage}: {count} keywords [{flag}]")

    reseed_needed = any(count < min_kw for count in coverage.values())
    logs.append(f"pool_inspection: coverage={coverage}, reseed_needed={reseed_needed}")

    if len(pool_df) < CONFIG["small_pool_threshold"]:
        warnings.append(
            f"pool_inspection: small pool ({len(pool_df)} < {CONFIG['small_pool_threshold']}) "
            "— HDBSCAN will be skipped, agglomerative used"
        )

    return {
        "pool_coverage":  coverage,
        "reseed_needed":  reseed_needed,
        "errors":         errors,
        "warnings":       warnings,
        "logs":           logs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 4: adaptive_reseed
# ─────────────────────────────────────────────────────────────────────────────

def adaptive_reseed(state: KeywordAgentState, config: RunnableConfig) -> dict:
    """
    Node 4: Targeted re-seed for stages below coverage threshold.
    Only processes stages where reseed_tracker[stage] < 1 (hard cap).
    Writes: pool_df, reseed_tracker
    """
    llm, _, googleads_client, customer_id = _clients(config)
    business_ctx   = state["business_context"]
    pool_df        = state["pool_df"].copy()
    pool_coverage  = state["pool_coverage"]
    reseed_tracker = dict(state.get("reseed_tracker", {}))
    logs    = list(state.get("logs", []))
    errors  = list(state.get("errors", []))
    warnings = list(state.get("warnings", []))
    min_kw  = CONFIG["inference"]["min_keywords_per_stage"]

    print(f"\n[Node: adaptive_reseed]")

    for stage, count in pool_coverage.items():
        if count >= min_kw:
            continue
        already_reseeded = reseed_tracker.get(stage, 0)
        if already_reseeded >= 1:
            warnings.append(
                f"adaptive_reseed: {stage} still sparse after 1 re-seed — "
                "flagging as market characteristic, not re-seeding again"
            )
            continue

        print(f"  [Reseed] Stage {stage} has {count} < {min_kw} keywords. Triggering targeted re-seed.")
        try:
            new_seeds = _generate_seeds(business_ctx, stage, llm, max_retries=2)
            print(f"  New seeds: {new_seeds}")

            new_df = expand_keywords(
                business_ctx     = business_ctx,
                stage            = f"{stage}_reseed",
                seeds            = new_seeds,
                googleads_client = googleads_client,
                customer_id      = customer_id,
                config           = CONFIG,
            )
            new_df["source_stage"] = stage

            # Save under reseed cache key
            reseed_path = os.path.join(
                CONFIG["outputs_dir"],
                f"{business_ctx['key']}_{stage.lower()}_reseed_raw.csv",
            )
            new_df.to_csv(reseed_path, index=False)

            # Merge — keep existing on conflict (dedup by keyword text)
            combined = pd.concat([pool_df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["keyword"], keep="first").reset_index(drop=True)
            pool_df  = combined

            reseed_tracker[stage] = already_reseeded + 1
            logs.append(
                f"adaptive_reseed: {stage} re-seeded → pool now {len(pool_df)} keywords"
            )
            print(f"  Pool after re-seed: {len(pool_df)} keywords")

        except Exception as exc:
            msg = f"adaptive_reseed: failed for stage {stage}: {exc}"
            errors.append(msg)
            print(f"    ERROR: {msg}")

    return {
        "pool_df":        pool_df,
        "reseed_tracker": reseed_tracker,
        "errors":         errors,
        "warnings":       warnings,
        "logs":           logs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 5: clustering
# ─────────────────────────────────────────────────────────────────────────────

def _score_keyword_for_intent(keyword: str) -> dict[str, float]:
    """Position-weighted signal-word scorer. Returns normalised probability vector."""
    kw_lower = keyword.lower()
    raw = {stage: 0.0 for stage in STAGES}
    for stage, stage_def in INTENT_TAXONOMY.items():
        for signal in stage_def["signal_words"]:
            if signal in kw_lower:
                pos    = kw_lower.find(signal)
                weight = 1.0 + (1.0 / (pos + 1))
                raw[stage] += weight
    total = sum(raw.values())
    if total == 0:
        return {s: 1.0 / len(STAGES) for s in STAGES}
    return {s: raw[s] / total for s in STAGES}


def _compute_intent_score_matrix(keywords: list[str]) -> np.ndarray:
    """Build (N, 3) intent score matrix — one row per keyword."""
    scores = np.array(
        [[_score_keyword_for_intent(kw)[stage] for stage in STAGES] for kw in keywords],
        dtype=np.float32,
    )
    return scores


def _build_composite_embedding(
    semantic_embeddings: np.ndarray,
    intent_scores: np.ndarray,
    weight: float,
) -> np.ndarray:
    """Concatenate semantic (768-dim) + intent (3-dim * weight), then L2-normalise."""
    intent_scaled = intent_scores * weight
    composite     = np.hstack([semantic_embeddings, intent_scaled])
    composite     = normalize(composite, norm="l2")
    return composite


def _run_hdbscan_with_rescue(X_5d: np.ndarray, pool_size: int) -> np.ndarray:
    """HDBSCAN + K-Means noise rescue. Falls back to agglomerative for small pools."""
    if pool_size < CONFIG["small_pool_threshold"]:
        print(f"  [Small pool branch] {pool_size} < {CONFIG['small_pool_threshold']} — using AgglomerativeClustering")
        agg = AgglomerativeClustering(
            n_clusters = CONFIG["agglomerative_n_clusters"],
            metric     = "euclidean",
            linkage    = "ward",
        )
        return agg.fit_predict(X_5d).astype(int)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size         = CONFIG["hdbscan_min_cluster_size"],
        min_samples              = CONFIG["hdbscan_min_samples"],
        cluster_selection_method = CONFIG["hdbscan_cluster_selection_method"],
        metric                   = CONFIG["hdbscan_metric"],
    )
    labels = clusterer.fit_predict(X_5d)

    n_noise    = (labels == -1).sum()
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"  HDBSCAN: {n_clusters} clusters, {n_noise} noise points ({n_noise/len(labels)*100:.1f}%)")

    if n_noise > 0 and n_clusters > 0:
        noise_mask   = labels == -1
        centroids    = np.array([
            X_5d[labels == c].mean(axis=0)
            for c in sorted(set(labels) - {-1})
        ])
        dists        = cdist(X_5d[noise_mask], centroids, metric="euclidean")
        nearest      = dists.argmin(axis=1)
        clean_labels = labels.copy()
        clean_labels[noise_mask] = np.array(list(sorted(set(labels) - {-1})))[nearest]
        print(f"  K-Means rescue: {n_noise} noise points assigned to nearest cluster centroids")
        return clean_labels

    return labels


def clustering(state: KeywordAgentState, config: RunnableConfig) -> dict:
    """
    Node 5: Composite embedding → UMAP 5D → HDBSCAN.
    Writes: embeddings, X_5d, X_2d, cluster_labels, silhouette_score
    """
    _, genai_client, _, _ = _clients(config)
    pool_df      = state["pool_df"]
    business_ctx = state["business_context"]
    logs    = list(state.get("logs", []))
    errors  = list(state.get("errors", []))
    warnings = list(state.get("warnings", []))

    print(f"\n[Node: clustering]")
    keywords = pool_df["keyword"].tolist()
    biz_key  = business_ctx["key"]

    try:
        os.makedirs(CONFIG.get("cache_dir", CONFIG["outputs_dir"]), exist_ok=True)

        # 1. Semantic embeddings
        semantic = embed_keywords(keywords, biz_key, genai_client, CONFIG)

        # 2. Intent score matrix
        intent_scores = _compute_intent_score_matrix(keywords)
        print(f"[Composite] Intent score matrix: {intent_scores.shape}")

        # 3. Composite embedding
        composite = _build_composite_embedding(semantic, intent_scores, CONFIG["intent_score_weight"])
        print(f"[Composite] Composite embedding: {composite.shape}")

        # 4. UMAP 5D — clustering pass
        print("[Composite] UMAP 5D...")
        reducer_5d = umap.UMAP(
            n_components = CONFIG["umap_composite_n_components"],
            n_neighbors  = CONFIG["umap_composite_n_neighbors"],
            min_dist     = CONFIG["umap_composite_min_dist"],
            metric       = CONFIG["umap_composite_metric"],
            random_state = 42,
        )
        X_5d = reducer_5d.fit_transform(composite)

        # 5. UMAP 2D — viz pass only (separate reducer, NEVER reuse 5D)
        print("[Composite] UMAP 2D viz pass...")
        reducer_2d = umap.UMAP(
            n_components = 2,
            n_neighbors  = CONFIG["umap_viz_n_neighbors"],
            min_dist     = CONFIG["umap_viz_min_dist"],
            metric       = CONFIG["umap_viz_metric"],
            random_state = 42,
        )
        X_2d = reducer_2d.fit_transform(composite)

        # 6. Cluster
        print(f"[Composite] Clustering (pool_size={len(pool_df)})...")
        labels = _run_hdbscan_with_rescue(X_5d, len(pool_df))
        n_clusters = len(set(labels))
        print(f"  Final cluster count: {n_clusters}")

        # 7. Silhouette score
        sil_score = None
        if n_clusters > 1:
            try:
                sil_score = float(sk_silhouette_score(X_5d, labels))
                print(f"  Silhouette score: {sil_score:.3f}")
                logs.append(f"clustering: silhouette={sil_score:.3f}, clusters={n_clusters}")
            except Exception as e:
                warnings.append(f"clustering: silhouette score failed: {e}")

    except Exception as exc:
        errors.append(f"clustering: {exc}")
        print(f"  ERROR: {exc}")
        return {"errors": errors, "warnings": warnings, "logs": logs}

    return {
        "embeddings":       semantic,
        "X_5d":             X_5d,
        "X_2d":             X_2d,
        "cluster_labels":   labels,
        "silhouette_score": sil_score,
        "errors":           errors,
        "warnings":         warnings,
        "logs":             logs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 6: post_clustering_ops
# ─────────────────────────────────────────────────────────────────────────────

GEO_STOPWORDS = {
    "near me", "nearby", "local", "in my area", "close by",
    "phoenix", "az", "los angeles", "la", "nyc", "new york",
    "chicago", "houston", "dallas", "miami", "seattle", "denver",
}


def _score_keyword(keyword: str) -> dict[str, float]:
    """Position-weighted signal-word scorer for a single keyword."""
    kw_lower = keyword.lower()
    raw = {stage: 0.0 for stage in STAGES}
    for stage, stage_def in INTENT_TAXONOMY.items():
        for signal in stage_def["signal_words"]:
            if signal in kw_lower:
                pos    = kw_lower.find(signal)
                weight = 1.0 + (1.0 / (pos + 1))
                raw[stage] += weight
    total = sum(raw.values())
    if total == 0:
        return {s: 1.0 / len(STAGES) for s in STAGES}
    return {s: raw[s] / total for s in STAGES}


def _detect_modifier_type(keywords: list[str]) -> Optional[str]:
    """Detect dominant modifier type from top keywords."""
    geo_pats   = [r"\bnear me\b", r"\bphoenix\b", r"\bla\b", r"\bnyc\b", r"\b[a-z]+ [a-z]{2}\b"]
    price_pats = [r"\bprice\b", r"\bcost\b", r"\baffordable\b", r"\bcheap\b", r"\bdiscount\b", r"\bcoupon\b"]
    comp_pats  = [r"\bvs\b", r"\bversus\b", r"\balternative\b", r"\bcompare\b", r"\bvs "]
    feat_pats  = [r"\bfeatures\b", r"\bintegration\b", r"\bwith\b", r"\bfor\b"]
    emrg_pats  = [r"\bemergency\b", r"\bsame day\b", r"\b24/7\b", r"\burgent\b"]
    brand_pats = [r"\breview\b", r"\bratings\b", r"\brated\b"]

    counts = {"geo": 0, "price": 0, "comparison": 0, "feature": 0, "emergency": 0, "brand": 0}
    for kw in keywords:
        kw_l = kw.lower()
        for pat in geo_pats:
            if re.search(pat, kw_l): counts["geo"] += 1
        for pat in price_pats:
            if re.search(pat, kw_l): counts["price"] += 1
        for pat in comp_pats:
            if re.search(pat, kw_l): counts["comparison"] += 1
        for pat in feat_pats:
            if re.search(pat, kw_l): counts["feature"] += 1
        for pat in emrg_pats:
            if re.search(pat, kw_l): counts["emergency"] += 1
        for pat in brand_pats:
            if re.search(pat, kw_l): counts["brand"] += 1

    dominant = max(counts, key=counts.get)
    return dominant if counts[dominant] > 0 else None


def _assign_intent_labels(df: pd.DataFrame, cluster_col: str = "cluster_id") -> pd.DataFrame:
    """Volume-weighted intent voting at cluster level."""
    df = df.copy()
    df["assigned_intent"]   = "UNKNOWN"
    df["intent_confidence"] = 0.0
    df["dominant_modifier"] = None

    for cluster_id in df[cluster_col].unique():
        mask     = df[cluster_col] == cluster_id
        cluster  = df[mask].copy()
        keywords = cluster["keyword"].tolist()
        volumes  = cluster["avg_monthly_searches"].fillna(0).values.astype(float)

        agg_scores = {stage: 0.0 for stage in STAGES}
        for kw, vol in zip(keywords, volumes):
            kw_scores = _score_keyword(kw)
            weight    = vol + 1.0
            for stage in STAGES:
                agg_scores[stage] += kw_scores[stage] * weight

        total = sum(agg_scores.values())
        if total == 0:
            intent     = "MIXED"
            confidence = 0.0
        else:
            normalised   = {s: agg_scores[s] / total for s in STAGES}
            top_stage    = max(normalised, key=normalised.get)
            top_score    = normalised[top_stage]
            second_score = sorted(normalised.values())[-2]
            if top_score < CONFIG["min_signal_weight"] or (top_score - second_score) < 0.15:
                intent     = "MIXED"
                confidence = top_score
            else:
                intent     = top_stage
                confidence = top_score

        modifier = _detect_modifier_type(keywords[:10])
        df.loc[mask, "assigned_intent"]   = intent
        df.loc[mask, "intent_confidence"] = confidence
        df.loc[mask, "dominant_modifier"] = modifier

    print(f"Intent labels assigned. Distribution:")
    print(df.drop_duplicates(subset=[cluster_col])["assigned_intent"].value_counts().to_string())
    return df


def _strip_geo_modifiers(keyword: str) -> str:
    words    = keyword.lower().split()
    filtered = [w for w in words if w not in GEO_STOPWORDS and len(w) > 1]
    return " ".join(filtered)


def _compute_base_phrase_overlap(kws_a: list[str], kws_b: list[str]) -> float:
    set_a = set(_strip_geo_modifiers(k) for k in kws_a if _strip_geo_modifiers(k))
    set_b = set(_strip_geo_modifiers(k) for k in kws_b if _strip_geo_modifiers(k))
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _deduplicate_geo_variants(df: pd.DataFrame, cluster_col: str = "cluster_id") -> pd.DataFrame:
    """Collapse clusters that are geo-variants of the same base ad group."""
    df          = df.copy()
    cluster_ids = sorted(df[cluster_col].unique())
    cluster_kws = {cid: df[df[cluster_col] == cid]["keyword"].tolist() for cid in cluster_ids}
    threshold   = CONFIG["geo_dedup_overlap_threshold"]
    merge_into  = {}

    for i, cid_a in enumerate(cluster_ids):
        if cid_a in merge_into:
            continue
        for cid_b in cluster_ids[i + 1:]:
            if cid_b in merge_into:
                continue
            a_geo = df[df[cluster_col] == cid_a]["dominant_modifier"].iloc[0] == "geo" if "dominant_modifier" in df.columns else False
            b_geo = df[df[cluster_col] == cid_b]["dominant_modifier"].iloc[0] == "geo" if "dominant_modifier" in df.columns else False
            if not (a_geo or b_geo):
                continue
            overlap = _compute_base_phrase_overlap(cluster_kws[cid_a], cluster_kws[cid_b])
            if overlap > threshold:
                vol_a = df[df[cluster_col] == cid_a]["avg_monthly_searches"].sum()
                vol_b = df[df[cluster_col] == cid_b]["avg_monthly_searches"].sum()
                keep, drop = (cid_a, cid_b) if vol_a >= vol_b else (cid_b, cid_a)
                merge_into[drop] = keep
                print(f"  Geo dedup: cluster {drop} → merged into cluster {keep} "
                      f"(base phrase overlap={overlap:.2f})")

    if merge_into:
        df[cluster_col] = df[cluster_col].map(lambda x: merge_into.get(x, x))
        print(f"  Total geo merges: {len(merge_into)} clusters collapsed")
    else:
        print("  No geo-variant clusters detected above threshold.")
    return df


def _merge_small_clusters(
    df: pd.DataFrame,
    X_embed: np.ndarray,
    cluster_col: str = "cluster_id",
) -> pd.DataFrame:
    """Merge clusters below min_keywords_per_ag into nearest valid cluster."""
    df      = df.copy()
    min_kw  = CONFIG["min_keywords_per_ag"]
    changed = True

    while changed:
        changed = False
        counts  = df[cluster_col].value_counts()
        small   = counts[counts < min_kw].index.tolist()
        if not small:
            break
        valid_ids = counts[counts >= min_kw].index.tolist()
        if not valid_ids:
            print("  WARNING: All clusters below min_keywords_per_ag — cannot merge further")
            break
        cid           = small[0]
        small_mask    = df[cluster_col] == cid
        small_indices = np.where(small_mask.values)[0]
        if len(small_indices) == 0:
            continue
        small_centroid  = X_embed[small_indices].mean(axis=0, keepdims=True)
        valid_centroids = np.array([X_embed[df[cluster_col] == vid].mean(axis=0) for vid in valid_ids])
        nearest_idx     = cdist(small_centroid, valid_centroids).argmin()
        target_id       = valid_ids[nearest_idx]
        df.loc[small_mask, cluster_col] = target_id
        print(f"  Merge: cluster {cid} ({counts[cid]} kw) → cluster {target_id} (below min={min_kw})")
        changed = True
    return df


def _split_large_clusters(
    df: pd.DataFrame,
    X_embed: np.ndarray,
    cluster_col: str = "cluster_id",
) -> pd.DataFrame:
    """Split clusters above max_keywords_per_ag via agglomerative re-clustering."""
    df      = df.copy()
    max_kw  = CONFIG["max_keywords_per_ag"]
    next_id = df[cluster_col].max() + 1

    for cid in df[cluster_col].unique():
        mask = df[cluster_col] == cid
        n    = mask.sum()
        if n <= max_kw:
            continue
        n_splits = max(2, int(np.ceil(n / (max_kw * 0.7))))
        print(f"  Split: cluster {cid} ({n} kw) → {n_splits} sub-clusters")
        indices = np.where(mask.values)[0]
        sub_X   = X_embed[indices]
        agg     = AgglomerativeClustering(n_clusters=n_splits, metric="euclidean", linkage="ward")
        sub_lbl = agg.fit_predict(sub_X)
        new_ids = list(range(next_id, next_id + n_splits))
        next_id += n_splits
        for local_id, global_id in enumerate(new_ids):
            sub_mask = sub_lbl == local_id
            df.loc[df.index[indices[sub_mask]], cluster_col] = global_id
    return df


def _apply_size_constraints(
    df: pd.DataFrame,
    X_embed: np.ndarray,
    cluster_col: str = "cluster_id",
) -> pd.DataFrame:
    print(f"\n[SizeConstraints] Before: {df[cluster_col].nunique()} clusters")
    df = _merge_small_clusters(df, X_embed, cluster_col)
    df = _split_large_clusters(df, X_embed, cluster_col)
    print(f"[SizeConstraints] After:  {df[cluster_col].nunique()} clusters")
    return df


def post_clustering_ops(state: KeywordAgentState, config: RunnableConfig) -> dict:
    """
    Node 6: Three post-clustering steps in strict order:
    1. Intent label assignment
    2. Geo-variant deduplication
    3. Cluster merge / split (size constraints)
    Writes: df_labelled
    """
    pool_df       = state["pool_df"].copy()
    X_5d          = state["X_5d"]
    cluster_labels = state["cluster_labels"]
    logs    = list(state.get("logs", []))
    errors  = list(state.get("errors", []))
    warnings = list(state.get("warnings", []))

    print(f"\n[Node: post_clustering_ops]")

    try:
        # Attach cluster labels to DataFrame
        pool_df["cluster_id"] = cluster_labels

        # Step 1: Intent label assignment
        df = _assign_intent_labels(pool_df, cluster_col="cluster_id")

        # Step 2: Geo-variant deduplication
        df = _deduplicate_geo_variants(df, cluster_col="cluster_id")

        # Step 3: Merge / split to enforce size constraints
        df = _apply_size_constraints(df, X_5d, cluster_col="cluster_id")

        logs.append(f"post_clustering_ops: {df['cluster_id'].nunique()} final clusters")

    except Exception as exc:
        errors.append(f"post_clustering_ops: {exc}")
        print(f"  ERROR: {exc}")
        return {"errors": errors, "warnings": warnings, "logs": logs}

    return {
        "df_labelled": df,
        "errors":      errors,
        "warnings":    warnings,
        "logs":        logs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 7: mixed_adjudication
# ─────────────────────────────────────────────────────────────────────────────

def _build_adjudication_prompt(
    cluster_id: int,
    top_keywords: list[str],
    candidate_intents: list[str],
) -> list:
    kw_str = "\n".join(f"  {i+1}. {kw}" for i, kw in enumerate(top_keywords[:5]))
    system_msg = SystemMessage(content=(
        "You are a search intent classification expert. "
        "You assign the single most appropriate funnel stage to a cluster of keywords. "
        "TOFU = informational, no purchase intent. "
        "MOFU = commercial investigation, comparing options. "
        "BOFU = transactional, ready to act. "
        "Be decisive — pick one, even if it's close."
    ))
    user_msg = HumanMessage(content=f"""Classify this keyword cluster.

Cluster ID: {cluster_id}
Keywords (top 5 by search volume):
{kw_str}

The deterministic classifier flagged this as MIXED — scoring was too close between:
{', '.join(candidate_intents)}

Assign exactly one intent stage: TOFU, MOFU, or BOFU.
Explain your reasoning in 1-2 sentences, citing which specific keywords drove your decision.
""")
    return [system_msg, user_msg]


def mixed_adjudication(state: KeywordAgentState, config: RunnableConfig) -> dict:
    """
    Node 7: LLM adjudication for MIXED clusters.
    Writes: df_final
    """
    llm, _, _, _ = _clients(config)
    df_labelled = state["df_labelled"].copy()
    logs    = list(state.get("logs", []))
    errors  = list(state.get("errors", []))
    warnings = list(state.get("warnings", []))

    print(f"\n[Node: mixed_adjudication]")

    df = df_labelled.copy()
    df["was_mixed"] = df["assigned_intent"] == "MIXED"
    mixed_ids = df[df["assigned_intent"] == "MIXED"]["cluster_id"].unique()

    if len(mixed_ids) == 0:
        print("[Adjudication] No MIXED clusters — skipping.")
        return {"df_final": df, "errors": errors, "warnings": warnings, "logs": logs}

    print(f"[Adjudication] {len(mixed_ids)} MIXED cluster(s) to adjudicate...")
    structured_llm = llm.with_structured_output(IntentAdjudication)

    for cid in mixed_ids:
        mask       = df["cluster_id"] == cid
        cluster_df = df[mask].sort_values("avg_monthly_searches", ascending=False)
        top_kws    = cluster_df["keyword"].head(5).tolist()

        stage_sums = {s: 0.0 for s in STAGES}
        for kw in cluster_df["keyword"].tolist():
            s = _score_keyword(kw)
            for st in STAGES:
                stage_sums[st] += s[st]
        sorted_stages = sorted(stage_sums, key=stage_sums.get, reverse=True)
        candidates    = sorted_stages[:2]

        try:
            messages = _build_adjudication_prompt(cid, top_kws, candidates)
            result   = structured_llm.invoke(messages)
            df.loc[mask, "assigned_intent"] = result.assigned_intent
            print(f"  Cluster {cid}: MIXED → {result.assigned_intent} "
                  f"({result.confidence}) | {result.reasoning[:80]}...")
            logs.append(f"mixed_adjudication: cluster {cid} → {result.assigned_intent} ({result.confidence})")
        except Exception as exc:
            warnings.append(
                f"mixed_adjudication: cluster {cid} adjudication failed ({exc}) — preserving MIXED"
            )
            print(f"  Cluster {cid}: adjudication failed ({exc}) — keeping MIXED for human review")

    return {
        "df_final": df,
        "errors":   errors,
        "warnings": warnings,
        "logs":     logs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 8: cluster_inference
# ─────────────────────────────────────────────────────────────────────────────

def _build_cluster_records(df: pd.DataFrame, cluster_col: str = "cluster_id") -> list[ClusterRecord]:
    """Build a ClusterRecord for each cluster from the labelled pool DataFrame."""
    records = []
    for cid in sorted(df[cluster_col].unique()):
        mask    = df[cluster_col] == cid
        cluster = df[mask].copy()
        by_vol  = cluster.sort_values("avg_monthly_searches", ascending=False)

        top_kws  = by_vol["keyword"].head(5).tolist()
        rep_kw   = top_kws[0] if top_kws else cluster["keyword"].iloc[0]
        intent   = cluster["assigned_intent"].mode()[0] if "assigned_intent" in cluster.columns else "UNKNOWN"
        modifier = cluster["dominant_modifier"].iloc[0] if "dominant_modifier" in cluster.columns else None

        total_vol = int(cluster["avg_monthly_searches"].sum())
        avg_vol   = float(cluster["avg_monthly_searches"].mean())
        avg_comp  = (
            float(cluster["competition_index"].mean())
            if "competition_index" in cluster.columns and cluster["competition_index"].notna().any()
            else None
        )

        records.append(ClusterRecord(
            cluster_id             = int(cid),
            assigned_intent        = intent,
            keyword_count          = len(cluster),
            total_monthly_volume   = total_vol,
            avg_monthly_volume     = round(avg_vol, 1),
            avg_competition_index  = round(avg_comp, 1) if avg_comp is not None else None,
            representative_keyword = rep_kw,
            top_keywords           = top_kws,
            dominant_modifier_type = modifier,
            ad_group_name          = rep_kw.title(),
        ))
    return records


def _run_cluster_inference(
    business_ctx: dict,
    df: pd.DataFrame,
    cluster_records: list[ClusterRecord],
) -> ClusterInferenceReport:
    """Compute the full ClusterInferenceReport — zero LLM calls, purely deterministic."""
    cfg = CONFIG["inference"]

    # 1. Volume-weighted intent distribution
    stage_volumes = {stage: 0 for stage in STAGES}
    total_volume  = 0
    for rec in cluster_records:
        if rec.assigned_intent in STAGES:
            stage_volumes[rec.assigned_intent] += rec.total_monthly_volume
            total_volume += rec.total_monthly_volume

    if total_volume > 0:
        intent_distribution = {s: round(stage_volumes[s] / total_volume, 3) for s in STAGES}
    else:
        intent_distribution = {s: 1.0 / len(STAGES) for s in STAGES}

    dominant_stage = max(intent_distribution, key=intent_distribution.get)
    dominant_share = intent_distribution[dominant_stage]

    # 2. Gap clusters — high volume AND low competition
    all_volumes = [rec.avg_monthly_volume for rec in cluster_records if rec.avg_monthly_volume > 0]
    pool_median = float(np.median(all_volumes)) if all_volumes else 0.0

    gap_clusters = []
    for rec in cluster_records:
        if rec.assigned_intent not in STAGES:
            continue
        is_high_vol = rec.avg_monthly_volume > pool_median * cfg["gap_volume_multiplier"]
        comp_val    = rec.avg_competition_index or 0
        is_low_comp = comp_val < cfg["gap_competition_max"]
        if is_high_vol and is_low_comp:
            gap_score = rec.avg_monthly_volume / (comp_val + 1)
            gap_clusters.append(GapCluster(
                cluster_id         = rec.cluster_id,
                ad_group_name      = rec.ad_group_name,
                assigned_intent    = rec.assigned_intent,
                avg_monthly_volume = rec.avg_monthly_volume,
                competition_index  = rec.avg_competition_index,
                gap_score          = round(gap_score, 2),
            ))

    gap_clusters.sort(key=lambda x: x.gap_score, reverse=True)

    # 3. Competitive pressure per stage
    stage_comp_vals = {stage: [] for stage in STAGES}
    for rec in cluster_records:
        if rec.assigned_intent in STAGES and rec.avg_competition_index is not None:
            stage_comp_vals[rec.assigned_intent].append(rec.avg_competition_index)

    competitive_pressure = {
        s: round(float(np.mean(v)), 1) if v else None
        for s, v in stage_comp_vals.items()
    }

    # 4. Missing stage detection
    stage_kw_counts = {stage: 0 for stage in STAGES}
    for rec in cluster_records:
        if rec.assigned_intent in STAGES:
            stage_kw_counts[rec.assigned_intent] += rec.keyword_count

    source_counts = df["source_stage"].value_counts().to_dict() if "source_stage" in df.columns else {}

    missing_stages = []
    for stage in STAGES:
        count = stage_kw_counts[stage]
        if count < cfg["min_keywords_per_stage"]:
            src_count = source_counts.get(stage, 0)
            if src_count < 5:
                diagnosis = "seeding_failure"
                note = (
                    f"Keyword Planner returned only {src_count} keywords from {stage} seeds "
                    "— likely seed quality issue"
                )
            else:
                diagnosis = "market_characteristic"
                note = (
                    f"Planner returned {src_count} keywords but all reclassified away from {stage} "
                    "— market may not have this intent phase"
                )
            missing_stages.append(MissingStageFlag(
                stage=stage, keyword_count=count, diagnosis=diagnosis, note=note
            ))

    # 5. Recommended focus — CONFIG-driven decision rule
    mofu_pressure = competitive_pressure.get("MOFU")
    bofu_pressure = competitive_pressure.get("BOFU")
    bofu_has_gaps = any(g.assigned_intent == "BOFU" for g in gap_clusters)

    if dominant_share >= cfg["dominant_stage_volume_share"]:
        recommended_focus = "single_stage"
        reasoning = (
            f"{dominant_stage} holds {dominant_share*100:.0f}% of total search volume "
            f"(threshold: >{cfg['dominant_stage_volume_share']*100:.0f}%). "
            f"Focus budget on {dominant_stage} — other stages don't have enough demand to justify split attention."
        )
    elif bofu_has_gaps:
        recommended_focus = "bofu_gap"
        top_gap = gap_clusters[0]
        reasoning = (
            f"BOFU gap cluster detected: '{top_gap.ad_group_name}' has avg {top_gap.avg_monthly_volume:.0f} "
            f"searches/month with competition_index={top_gap.competition_index} "
            f"(threshold: <{cfg['gap_competition_max']}). Lead with BOFU gap exploitation."
        )
    elif (mofu_pressure is not None and bofu_pressure is not None
          and mofu_pressure > cfg["mofu_pressure_high_threshold"]
          and bofu_pressure < cfg["bofu_pressure_low_threshold"]):
        recommended_focus = "flank"
        reasoning = (
            f"High MOFU competitive pressure ({mofu_pressure:.1f} > {cfg['mofu_pressure_high_threshold']}) "
            f"but low BOFU pressure ({bofu_pressure:.1f} < {cfg['bofu_pressure_low_threshold']}). "
            "Avoid head-on MOFU competition — flank via BOFU gap clusters instead."
        )
    else:
        recommended_focus = "balanced"
        reasoning = (
            f"No single dominant stage ({dominant_share*100:.0f}% < {cfg['dominant_stage_volume_share']*100:.0f}%), "
            "no high-priority gap clusters, competitive pressure roughly even across stages. "
            "Balanced full-funnel allocation."
        )

    return ClusterInferenceReport(
        business_key                = business_ctx["key"],
        intent_distribution         = intent_distribution,
        dominant_stage              = dominant_stage,
        dominant_stage_volume_share = dominant_share,
        gap_clusters                = gap_clusters,
        competitive_pressure        = competitive_pressure,
        missing_stages              = missing_stages,
        recommended_focus           = recommended_focus,
        recommended_focus_reasoning = reasoning,
    )


def cluster_inference(state: KeywordAgentState, config: RunnableConfig) -> dict:
    """
    Node 8: Deterministic cluster inference. Zero LLM calls.
    Writes: cluster_records, inference_report
    """
    business_ctx = state["business_context"]
    df_final     = state["df_final"]
    logs    = list(state.get("logs", []))
    errors  = list(state.get("errors", []))
    warnings = list(state.get("warnings", []))

    print(f"\n[Node: cluster_inference]")

    try:
        records = _build_cluster_records(df_final)
        report  = _run_cluster_inference(business_ctx, df_final, records)

        print(f"  Dominant stage:    {report.dominant_stage} ({report.dominant_stage_volume_share*100:.1f}%)")
        print(f"  Gap clusters:      {len(report.gap_clusters)}")
        print(f"  Recommended focus: {report.recommended_focus.upper()}")
        logs.append(
            f"cluster_inference: {len(records)} clusters, "
            f"focus={report.recommended_focus}"
        )

    except Exception as exc:
        errors.append(f"cluster_inference: {exc}")
        print(f"  ERROR: {exc}")
        return {"errors": errors, "warnings": warnings, "logs": logs}

    return {
        "cluster_records":  records,
        "inference_report": report,
        "errors":           errors,
        "warnings":         warnings,
        "logs":             logs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 9: adgroup_context_build
# ─────────────────────────────────────────────────────────────────────────────

def _get_competition_level(index: Optional[float]) -> str:
    if index is None:
        return "UNKNOWN"
    for (low, high), label in COMPETITION_LEVEL_MAP.items():
        if low <= index < high:
            return label
    return "HIGH"


def _derive_topic_from_top_keywords(top_keywords: list[str]) -> str:
    """
    Derive a clean topic from top-3 keywords by finding the common thread.
    Picks the keyword with fewest words (most specific without qualifiers),
    then strips intent signal words and title-cases.
    """
    candidates = sorted(top_keywords[:3], key=lambda k: len(k.split()))
    all_signal = set()
    for stage_def in INTENT_TAXONOMY.values():
        all_signal.update(stage_def["signal_words"])

    for kw in candidates:
        words    = kw.lower().split()
        stripped = [w for w in words if w not in all_signal and len(w) > 2]
        if len(stripped) >= 2:
            return " ".join(stripped).title()

    return candidates[0].title()


def _build_campaign_name(business_name: str, intent_stage: str) -> str:
    stage_names = {"TOFU": "Awareness", "MOFU": "Consideration", "BOFU": "Conversion"}
    return f"{business_name} — {stage_names.get(intent_stage, intent_stage)}"


def _build_adgroup_contexts(
    business_ctx: dict,
    cluster_records: list[ClusterRecord],
) -> list[AdGroupContext]:
    """Build one AdGroupContext per cluster."""
    contexts = []
    for rec in cluster_records:
        stage      = rec.assigned_intent if rec.assigned_intent in STAGES else "MOFU"
        comp_level = _get_competition_level(rec.avg_competition_index)
        topic      = _derive_topic_from_top_keywords(rec.top_keywords)
        pain_point = MODIFIER_PAIN_POINT.get(rec.dominant_modifier_type, MODIFIER_PAIN_POINT[None])

        user_goal_template = INTENT_TAXONOMY[stage]["user_goal"]
        user_goal          = f"{user_goal_template} Searching for: {topic.lower()}."

        ctx = AdGroupContext(
            ad_group_name          = rec.ad_group_name,
            campaign_name          = _build_campaign_name(business_ctx["name"], stage),
            intent_stage           = rec.assigned_intent,   # preserve MIXED for transparency
            cluster_id             = rec.cluster_id,
            representative_keyword = rec.representative_keyword,
            top_keywords           = rec.top_keywords,
            avg_monthly_volume     = rec.avg_monthly_volume,
            total_monthly_volume   = rec.total_monthly_volume,
            competition_level      = comp_level,
            dominant_modifier_type = rec.dominant_modifier_type,
            topic                  = topic,
            user_goal              = user_goal,
            pain_point             = pain_point,
            business_name          = business_ctx["name"],
            relevant_service       = business_ctx["services"][0],
            key_differentiator     = business_ctx["key_differentiator"],
            landing_page_url       = business_ctx["landing_page_url"],
        )
        contexts.append(ctx)
    return contexts


def _build_keyword_pool_map(
    business_ctx: dict,
    df_final: pd.DataFrame,
    cluster_records: list[ClusterRecord],
    silhouette_score: Optional[float],
) -> KeywordPoolMap:
    """Build the KeywordPoolMap for Strategy Agent and Bidding Agent consumption."""
    pools: dict[str, list[str]] = {}
    for stage in STAGES:
        if "assigned_intent" in df_final.columns:
            pools[stage] = df_final[df_final["assigned_intent"] == stage]["keyword"].tolist()
        else:
            pools[stage] = []

    return KeywordPoolMap(
        business_key    = business_ctx["key"],
        business_name   = business_ctx["name"],
        total_keywords  = len(df_final),
        pools           = pools,
        clusters        = cluster_records,
        silhouette_score = silhouette_score,
        metadata        = {
            "clustering_method": "composite",
            "total_clusters":    len(cluster_records),
        },
    )


def adgroup_context_build(state: KeywordAgentState, config: RunnableConfig) -> dict:
    """
    Node 9: Build AdGroupContext list and KeywordPoolMap.
    Writes: adgroup_contexts, keyword_pool_map
    """
    business_ctx    = state["business_context"]
    cluster_records = state["cluster_records"]
    df_final        = state["df_final"]
    silhouette_sc   = state.get("silhouette_score")
    logs    = list(state.get("logs", []))
    errors  = list(state.get("errors", []))
    warnings = list(state.get("warnings", []))

    print(f"\n[Node: adgroup_context_build]")

    try:
        adgroup_contexts = _build_adgroup_contexts(business_ctx, cluster_records)
        keyword_pool_map = _build_keyword_pool_map(
            business_ctx, df_final, cluster_records, silhouette_sc
        )

        # Serialise adgroup_contexts to outputs
        os.makedirs(CONFIG["outputs_dir"], exist_ok=True)
        output_path = os.path.join(
            CONFIG["outputs_dir"],
            f"{business_ctx['key']}_adgroup_contexts.json",
        )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                [ctx.model_dump() for ctx in adgroup_contexts],
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"  {len(adgroup_contexts)} AdGroupContext objects → {output_path}")
        logs.append(
            f"adgroup_context_build: {len(adgroup_contexts)} ad groups, saved to {output_path}"
        )

    except Exception as exc:
        errors.append(f"adgroup_context_build: {exc}")
        print(f"  ERROR: {exc}")
        return {"errors": errors, "warnings": warnings, "logs": logs}

    return {
        "adgroup_contexts": adgroup_contexts,
        "keyword_pool_map": keyword_pool_map,
        "errors":           errors,
        "warnings":         warnings,
        "logs":             logs,
    }
