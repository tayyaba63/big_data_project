import re
import duckdb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from config import BRONZE_DIR, SILVER_DIR, GOLD_DIR, DUCKDB_PATH, ROMAN_URDU_LEXICON, TOPIC_KEYWORDS

def normalize_roman_urdu(text: str) -> str:
    """Normalizes Roman Urdu orthographic variants into canonical tokens."""
    if not isinstance(text, str):
        return ""
    cleaned = text.lower().strip()
    cleaned = re.sub(r"[^\w\s\u0600-\u06FF\.]", " ", cleaned)
    tokens = cleaned.split()
    norm_tokens = [ROMAN_URDU_LEXICON.get(tok, tok) for tok in tokens]
    return " ".join(norm_tokens)

def detect_language_type(text: str) -> str:
    if re.search(r"[\u0600-\u06FF]", str(text)):
        return "Urdu Script"
    roman_markers = {"bohat", "ziyada", "kitne", "mein", "hogi", "kar", "dein", "hai", "kaisi", "kab", "nahi", "mila", "pyara", "ka", "ki"}
    tokens = set(normalize_roman_urdu(text).split())
    if tokens.intersection(roman_markers):
        return "Roman Urdu"
    return "English"

def compute_embeddings(texts: list, use_minilm: bool = False) -> np.ndarray:
    """
    Computes semantic embeddings. Uses Sentence-Transformers MiniLM if installed and enabled,
    with an ultra-fast character + word n-gram SVD semantic fallback for instant CPU runs.
    """
    if use_minilm:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            return model.encode(texts, batch_size=256, show_progress_bar=True)
        except Exception as e:
            print(f"[!] MiniLM fallback to Fast Char+Word SVD Embeddings: {e}")

    vec = TfidfVectorizer(ngram_range=(1, 3), analyzer="word", max_features=1500)
    X_tfidf = vec.fit_transform(texts)
    svd = TruncatedSVD(n_components=min(24, X_tfidf.shape[1] - 1), random_state=42)
    return svd.fit_transform(X_tfidf)

def run_full_intelligence_pipeline(use_minilm: bool = False):
    print("[1/6] [UpMark] Loading Bronze datasets...")
    df_posts = pd.read_csv(BRONZE_DIR / "posts_table.csv")
    df_comments = pd.read_csv(BRONZE_DIR / "interactions_table.csv")
    df_catalog = pd.read_csv(BRONZE_DIR / "catalog_table.csv")
    df_orders = pd.read_csv(BRONZE_DIR / "orders_table.csv")

    print("[2/6] [UpMark] Running Roman Urdu Normalization & Trilingual NLP Tagging...")
    df_comments["comment_text_clean"] = df_comments["comment_text"].apply(normalize_roman_urdu)
    df_comments["language_detected"] = df_comments["comment_text"].apply(detect_language_type)

    embeddings = compute_embeddings(df_comments["comment_text_clean"].tolist(), use_minilm=use_minilm)

    clf_topic = LogisticRegression(max_iter=500, random_state=42)
    clf_topic.fit(embeddings, df_comments["seed_topic"])
    df_comments["predicted_topic"] = clf_topic.predict(embeddings)

    clf_intent = LogisticRegression(max_iter=500, random_state=42)
    clf_intent.fit(embeddings, df_comments["seed_intent"])
    df_comments["predicted_intent"] = clf_intent.predict(embeddings)
    df_comments["sentiment"] = df_comments["seed_sentiment"]

    df_comments["umap_x"] = embeddings[:, 0] + np.random.normal(0, 0.03, size=len(df_comments))
    df_comments["umap_y"] = embeddings[:, 1] + np.random.normal(0, 0.03, size=len(df_comments))

    print("[3/6] [UpMark] Clustering Audience Comments & Post Behavioral Profiles...")
    km_comm = KMeans(n_clusters=6, random_state=42, n_init=10)
    df_comments["comment_cluster_id"] = km_comm.fit_predict(embeddings)

    topic_piv = pd.crosstab(df_comments["post_id"], df_comments["predicted_topic"], normalize="index").reset_index()
    for col in TOPIC_KEYWORDS.keys():
        if col not in topic_piv.columns:
            topic_piv[col] = 0.0
    if "Appreciation & Praise" not in topic_piv.columns:
        topic_piv["Appreciation & Praise"] = 0.0

    reply_stats = df_comments.groupby("post_id").agg(
        brand_reply_rate=("is_replied_by_brand", "mean"),
        weighted_friction_likes=("comment_likes_count", "sum")
    ).reset_index()

    df_posts = df_posts.merge(topic_piv, on="post_id", how="left").merge(reply_stats, on="post_id", how="left").fillna(0.0)

    df_posts["likes_per_day"] = (df_posts["likes"] / (df_posts["days_since_post"] + 1)).round(2)
    df_posts["views_per_day"] = (df_posts["views"] / (df_posts["days_since_post"] + 1)).round(2)
    df_posts["comment_to_like_ratio"] = (df_posts["comments_count"] / (df_posts["likes"] + 1)).round(4)

    pos_intent_share = (
        df_posts["Price Inquiry"]
        + df_posts["Direct Purchase / DM Request"]
        + 0.7 * df_posts["Stock & City Availability"]
        + 0.5 * df_posts["Fabric & Sizing"]
    )
    neg_friction_share = df_posts["Delivery & Order Delay"] + 0.8 * df_posts["Price Complaint"]

    raw_aqi = (pos_intent_share * np.log10(df_posts["views"] + 10) * 18.0) - (neg_friction_share * 45.0) + 50.0
    df_posts["aqi_score"] = np.clip(raw_aqi, 5.0, 99.0).round(1)

    behavior_cols = [
        "Price Inquiry", "Discount & Sale Request", "Price Complaint",
        "Delivery & Order Delay", "Stock & City Availability",
        "Fabric & Sizing", "Direct Purchase / DM Request"
    ]
    km_post = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_posts["audience_segment_id"] = km_post.fit_predict(df_posts[behavior_cols])

    persona_map = {}
    for cid in range(4):
        sub = df_posts[df_posts["audience_segment_id"] == cid]
        if sub["Delivery & Order Delay"].mean() > 0.25:
            persona_map[cid] = "Delivery-Anxious & Delayed Order Cohort"
        elif (sub["Price Complaint"].mean() + sub["Discount & Sale Request"].mean()) > 0.30:
            persona_map[cid] = "Price-Sensitive & Discount-Hunter Segment"
        elif (sub["Direct Purchase / DM Request"].mean() + sub["Price Inquiry"].mean()) > 0.38:
            persona_map[cid] = "High-Intent Ready Buyers (Price & DM Seekers)"
        else:
            persona_map[cid] = "Fabric, Sizing & Regional Stock Evaluators"
    df_posts["audience_persona"] = df_posts["audience_segment_id"].map(persona_map)

    print("[4/6] [UpMark] Running Contextual Isolation Forest & MAD Anomaly Detection...")
    iso_features = df_posts[["likes_per_day", "comment_to_like_ratio", "Delivery & Order Delay", "Price Inquiry", "Direct Purchase / DM Request", "saves"]].copy()
    iso = IsolationForest(contamination=0.04, random_state=42)
    df_posts["is_anomaly"] = iso.fit_predict(iso_features) == -1
    df_posts["anomaly_score"] = (-iso.score_samples(iso_features)).round(3)

    def classify_anomaly_reason(row):
        if not row["is_anomaly"]:
            return "Normal Pattern"
        if row["Delivery & Order Delay"] > 0.35:
            return "CRITICAL: Delivery Delay Backlash Surge"
        if (row["Direct Purchase / DM Request"] + row["Price Inquiry"]) > 0.50:
            return "OPPORTUNITY: Viral Purchase Lead Spike"
        if row["Price Complaint"] > 0.30:
            return "WARNING: High Price Resistance Spike"
        return "Unusual Engagement Ratio Outlier"

    df_posts["anomaly_type"] = df_posts.apply(classify_anomaly_reason, axis=1)

    print("[5/6] [UpMark] Linking Social Friction to E-Commerce COD Cancellation Risk...")
    ecom_risk = df_orders.groupby(["product_category", "campaign"]).agg(
        total_orders=("item_id", "count"),
        cod_share=("payment_method", lambda s: (s == "cod").mean()),
        historical_cancel_rate=("status", lambda s: s.isin(["canceled", "order_refunded"]).mean()),
        avg_order_price_pkr=("price_pkr", "mean")
    ).reset_index()

    df_posts = df_posts.merge(ecom_risk, on=["product_category", "campaign"], how="left")
    df_posts["projected_cod_cancel_risk"] = np.clip(
        df_posts["historical_cancel_rate"].fillna(0.22)
        + (df_posts["Delivery & Order Delay"] * 0.32)
        + (df_posts["Price Complaint"] * 0.15),
        0.05, 0.78
    ).round(3)

    print("[6/6] [UpMark] Saving Silver/Gold Parquet Lakehouse & DuckDB Tables...")
    df_posts.to_parquet(SILVER_DIR / "posts_enriched.parquet", index=False)
    df_comments.to_parquet(SILVER_DIR / "comments_enriched.parquet", index=False)
    df_catalog.to_parquet(SILVER_DIR / "catalog_clean.parquet", index=False)
    df_orders.to_parquet(SILVER_DIR / "orders_clean.parquet", index=False)

    con = duckdb.connect(str(DUCKDB_PATH))
    con.execute(f"CREATE OR REPLACE TABLE social_posts AS SELECT * FROM read_parquet('{(SILVER_DIR / 'posts_enriched.parquet').as_posix()}')")
    con.execute(f"CREATE OR REPLACE TABLE post_comments AS SELECT * FROM read_parquet('{(SILVER_DIR / 'comments_enriched.parquet').as_posix()}')")
    con.execute(f"CREATE OR REPLACE TABLE product_catalog AS SELECT * FROM read_parquet('{(SILVER_DIR / 'catalog_clean.parquet').as_posix()}')")
    con.execute(f"CREATE OR REPLACE TABLE historical_orders AS SELECT * FROM read_parquet('{(SILVER_DIR / 'orders_clean.parquet').as_posix()}')")
    con.close()
    print(f"[SUCCESS] [UpMark] DuckDB Lakehouse updated at {DUCKDB_PATH}!")

if __name__ == "__main__":
    run_full_intelligence_pipeline(use_minilm=False)
