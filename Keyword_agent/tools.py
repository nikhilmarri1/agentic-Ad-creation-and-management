"""
Pure I/O functions with no graph logic.
Clients and config are injected — never imported as globals.
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from google.genai import types
from google.protobuf.json_format import MessageToDict
from sklearn.preprocessing import normalize


def embed_keywords(
    keywords: list[str],
    biz_key: str,
    genai_client,
    config: dict,
) -> np.ndarray:
    """
    Batch embed via gemini-embedding-001. L2-normalise before return.
    Cache to outputs/cache/{biz_key}_embeddings.npy.
    task_type = "CLUSTERING".

    Args:
        keywords:     list of keyword strings to embed
        biz_key:      business key used for cache filename
        genai_client: google.genai.Client instance
        config:       CONFIG dict from config.py

    Returns:
        (N, 768) L2-normalised float32 numpy array
    """
    cache_dir  = config.get("cache_dir", config["outputs_dir"])
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{biz_key}_embeddings.npy")

    if os.path.exists(cache_path):
        embeddings = np.load(cache_path)
        print(f"[Embeddings] Cache hit: {cache_path} — shape {embeddings.shape}")
        return embeddings

    print(f"[Embeddings] Embedding {len(keywords)} keywords...")
    batch_size       = config["embedding_batch_size"]
    rpm              = config.get("embedding_rpm", 100)
    quota_per_window = rpm - 1    # max keywords per 60-second window
    sent_in_window   = 0
    all_embeddings   = []

    n_batches = (len(keywords) + batch_size - 1) // batch_size
    for batch_idx in range(n_batches):
        i     = batch_idx * batch_size
        batch = keywords[i: i + batch_size]

        # Pause if this batch would exceed the per-minute keyword quota
        if sent_in_window + len(batch) > quota_per_window:
            print(f"\n  [Rate limit] {sent_in_window} keywords sent — waiting 60 s for quota reset...")
            time.sleep(60)
            sent_in_window = 0

        resp = genai_client.models.embed_content(
            model    = config["embedding_model"],
            contents = batch,
            config   = types.EmbedContentConfig(task_type="CLUSTERING"),
        )
        all_embeddings.extend([e.values for e in resp.embeddings])
        sent_in_window += len(batch)
        time.sleep(0.3)
        print(f"  Batch {batch_idx + 1}/{n_batches} done", end="\r")

    embeddings = np.array(all_embeddings, dtype=np.float32)
    embeddings = normalize(embeddings, norm="l2")  # L2-normalise
    np.save(cache_path, embeddings)
    print(f"\n[Embeddings] Saved: {cache_path} — shape {embeddings.shape}")
    return embeddings


def expand_keywords(
    business_ctx: dict,
    stage: str,
    seeds: list[str],
    googleads_client,
    customer_id: str,
    config: dict,
) -> pd.DataFrame:
    """
    Single KeywordPlanIdeaService call. Cache to outputs/{biz_key}_{stage}_raw.csv.
    Returns DataFrame with all metric columns + source_stage tag.

    Args:
        business_ctx:      business context dict (must have 'key', 'language_resource',
                           'geo_target_resource')
        stage:             funnel stage string used for cache key and source_stage tag
        seeds:             list of seed keyword strings
        googleads_client:  GoogleAdsClient instance
        customer_id:       Google Ads customer ID (no dashes)
        config:            CONFIG dict from config.py

    Returns:
        pd.DataFrame with columns: keyword, avg_monthly_searches, competition,
        competition_index, low_top_of_page_bid_micros, high_top_of_page_bid_micros,
        source_stage
    """
    biz_key    = business_ctx["key"]
    cache_path = os.path.join(config["outputs_dir"], f"{biz_key}_{stage.lower()}_raw.csv")

    if os.path.exists(cache_path):
        print(f"    [cache hit] {stage}: {cache_path}")
        return pd.read_csv(cache_path)

    svc     = googleads_client.get_service("KeywordPlanIdeaService")
    request = googleads_client.get_type("GenerateKeywordIdeasRequest")
    request.customer_id = customer_id
    request.language    = business_ctx["language_resource"]
    request.geo_target_constants.append(business_ctx["geo_target_resource"])
    request.include_adult_keywords = False
    request.keyword_plan_network   = (
        googleads_client.enums.KeywordPlanNetworkEnum.GOOGLE_SEARCH_AND_PARTNERS
    )
    request.page_size = config["max_results_per_call"]

    kw_seed = googleads_client.get_type("KeywordSeed")
    kw_seed.keywords.extend(seeds)
    request.keyword_seed = kw_seed

    response      = svc.generate_keyword_ideas(request=request)
    response_dict = MessageToDict(response._pb)

    rows = []
    for item in response_dict.get("results", [])[:config["max_results_per_call"]]:
        metrics = item.get("keywordIdeaMetrics", {}) or {}
        rows.append({
            "keyword":                     item.get("text"),
            "avg_monthly_searches":        int(metrics["avgMonthlySearches"]) if metrics.get("avgMonthlySearches") else None,
            "competition":                 metrics.get("competition", "UNKNOWN"),
            "competition_index":           float(metrics["competitionIndex"]) if metrics.get("competitionIndex") else None,
            "low_top_of_page_bid_micros":  int(metrics["lowTopOfPageBidMicros"]) if metrics.get("lowTopOfPageBidMicros") else None,
            "high_top_of_page_bid_micros": int(metrics["highTopOfPageBidMicros"]) if metrics.get("highTopOfPageBidMicros") else None,
            "source_stage": stage,
        })

    df = pd.DataFrame(rows)
    df.to_csv(cache_path, index=False)
    print(f"    {stage}: {len(df)} keywords → {cache_path}")
    return df
