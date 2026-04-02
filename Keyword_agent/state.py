from typing import TypedDict, Optional


class KeywordAgentState(TypedDict):
    # ── Input (provided before graph runs) ──────────────────────────────────
    business_context:       dict                    # name, key, industry, services, etc.
    geo_targeting_config:   dict                    # from upstream Geo Node

    # ── Phase 1: Seed generation ─────────────────────────────────────────────
    seeds_by_stage:         Optional[dict]          # {stage: [keyword strings]}

    # ── Phase 2: Expansion ───────────────────────────────────────────────────
    pool_df:                Optional[object]        # pd.DataFrame — keyword pool
    pool_coverage:          Optional[dict]          # {stage: int count}

    # ── Phase 3: Re-seed control ─────────────────────────────────────────────
    reseed_tracker:         dict                    # {stage: int passes used}
    reseed_needed:          bool

    # ── Phase 4: Clustering ──────────────────────────────────────────────────
    embeddings:             Optional[object]        # np.ndarray (N, 768) semantic
    X_5d:                   Optional[object]        # np.ndarray — composite UMAP 5D
    X_2d:                   Optional[object]        # np.ndarray — UMAP 2D viz only
    cluster_labels:         Optional[object]        # np.ndarray — composite labels
    silhouette_score:       Optional[float]

    # ── Phase 5: Post-clustering (single node) ───────────────────────────────
    df_labelled:            Optional[object]        # pd.DataFrame + assigned_intent col

    # ── Phase 6: MIXED adjudication ──────────────────────────────────────────
    df_final:               Optional[object]        # pd.DataFrame — fully labelled

    # ── Phase 7: Inference ───────────────────────────────────────────────────
    cluster_records:        Optional[list]          # list[ClusterRecord]
    inference_report:       Optional[object]        # ClusterInferenceReport

    # ── Phase 8: Output boundary ─────────────────────────────────────────────
    adgroup_contexts:       Optional[list]          # list[AdGroupContext]
    keyword_pool_map:       Optional[object]        # KeywordPoolMap

    # ── Control ──────────────────────────────────────────────────────────────
    errors:                 list[str]
    warnings:               list[str]
    logs:                   list[str]
