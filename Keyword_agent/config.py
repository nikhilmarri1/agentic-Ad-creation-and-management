# All tunable parameters in one place.
# Never hardcode these values in nodes.py or tools.py.

CONFIG = {
    # LLM
    "llm_model":           "llama-3.3-70b-versatile",
    "llm_temperature":     0.3,

    # Embeddings
    "embedding_model":        "models/gemini-embedding-001",
    "embedding_batch_size":   10,
    "embedding_rpm":          100,   # free-tier limit; chunk requests to stay under this
    "intent_score_weight":    0.3,   # relative weight of 3-dim intent vector in composite embed
                                     # 0.0 = pure semantic, 1.0 = pure intent, 0.3 is a good start

    # Keyword Planner
    "seeds_per_stage":          8,
    "max_results_per_call":     50,

    # ── Composite embedding UMAP params (clustering pass) ─────────────────
    "umap_composite_n_components":   5,
    "umap_composite_min_dist":       0.0,
    "umap_composite_n_neighbors":    10,
    "umap_composite_metric":         "cosine",

    # ── Two-stage UMAP params (Approach B — not used in production graph) ─
    "umap_topic_n_components":       5,
    "umap_topic_n_neighbors":        25,
    "umap_topic_min_dist":           0.0,
    "umap_intent_n_components":      5,
    "umap_intent_n_neighbors":       6,
    "umap_intent_min_dist":          0.0,

    # ── Shared viz pass (NEVER feed into HDBSCAN) ─────────────────────────
    "umap_viz_n_components":   2,
    "umap_viz_min_dist":       0.1,
    "umap_viz_n_neighbors":    10,
    "umap_viz_metric":         "cosine",

    # HDBSCAN
    "hdbscan_min_cluster_size":          5,
    "hdbscan_min_samples":               3,
    "hdbscan_cluster_selection_method":  "eom",
    "hdbscan_metric":                    "euclidean",

    # K-Means noise rescue
    "kmeans_noise_rescue_k":   5,

    # Small pool guard: if pool_size < this, skip HDBSCAN → agglomerative
    "small_pool_threshold":     80,
    "agglomerative_n_clusters": 6,

    # Ad group size constraints (post-clustering merge/split)
    "min_keywords_per_ag":   3,
    "max_keywords_per_ag":   20,

    # Geo-variant deduplication threshold
    "geo_dedup_overlap_threshold": 0.6,   # >60% base phrase overlap → collapse

    # Intent labelling
    "min_signal_weight":   0.3,   # cluster must score > this to get a label (not MIXED)

    # ── Cluster Inference Node thresholds (CONFIG-driven) ─────────────────
    "inference": {
        "dominant_stage_volume_share":    0.60,   # volume share above this → single-stage focus
        "gap_volume_multiplier":          1.2,    # cluster avg_volume > pool_median * this
        "gap_competition_max":            40.0,   # competition_index below this → low pressure
        "min_keywords_per_stage":         5,      # below this → stage flagged as missing
        "mofu_pressure_high_threshold":   50.0,   # mean competition_index above this = HIGH MOFU
        "bofu_pressure_low_threshold":    35.0,   # mean competition_index below this = LOW BOFU
    },

    # Paths
    "googleads_yaml_path":  "../google-ads.yaml",
    "outputs_dir":          "outputs",
    "cache_dir":            "outputs/cache",
}


INTENT_TAXONOMY = {
    "TOFU": {
        "label": "TOFU",
        "name": "Informational",
        "user_goal": "User is learning and seeking information. No purchase intent.",
        "signal_words": [
            "what is", "how to", "guide", "tutorial", "tips", "explained",
            "difference between", "overview", "learn", "understand", "benefits of",
            "introduction", "basics", "101", "definition", "why", "when to",
        ],
        "campaign_implication": "Awareness campaigns — lower bids, educational copy, ROAS expectations lower",
    },
    "MOFU": {
        "label": "MOFU",
        "name": "Commercial Investigation",
        "user_goal": "User is evaluating options and comparing solutions. Considering purchase.",
        "signal_words": [
            "best", "vs", "versus", "compare", "alternative", "review",
            "alternatives to", "pricing", "worth it", "pros and cons", "top",
            "comparison", "recommend", "which is better", "features",
            "ranked", "ratings", "cost of", "affordable", "cheap",
        ],
        "campaign_implication": "Consideration campaigns — medium-high bids, feature-focused differentiation copy",
    },
    "BOFU": {
        "label": "BOFU",
        "name": "Transactional",
        "user_goal": "User is ready to act — buy, sign up, hire, or contact.",
        "signal_words": [
            "buy", "get", "sign up", "free trial", "demo", "quote", "hire",
            "near me", "cost", "price", "coupon", "discount", "book",
            "schedule", "install", "repair", "emergency", "same day",
            "purchase", "order", "deal", "service", "now", "today",
        ],
        "campaign_implication": "Conversion campaigns — highest bids, direct response copy, strong CTA",
    },
}

STAGES = list(INTENT_TAXONOMY.keys())   # ["TOFU", "MOFU", "BOFU"]

MODIFIER_PAIN_POINT = {
    "geo":         "finding a nearby, trustworthy provider",
    "price":       "uncertain about cost or value for money",
    "comparison":  "unsure which option best fits their needs",
    "feature":     "evaluating whether the product meets their requirements",
    "emergency":   "urgent need, time pressure, low tolerance for friction",
    "brand":       "evaluating a specific brand before committing",
    None:          "seeking the best solution for their problem",
}

COMPETITION_LEVEL_MAP = {
    (0,   30):  "LOW",
    (30,  60):  "MEDIUM",
    (60,  101): "HIGH",
}
