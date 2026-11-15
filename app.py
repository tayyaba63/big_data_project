import streamlit as st
import pandas as pd
import duckdb
import plotly.express as px
from config import DUCKDB_PATH, GOLD_DIR

st.set_page_config(
    page_title="UpMark | Social & Campaign Intelligence",
    page_icon="\U0001f4c8",
    layout="wide"
)

@st.cache_data
def load_tables():
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    posts = con.execute("SELECT * FROM social_posts").df()
    comments = con.execute("SELECT * FROM post_comments").df()
    catalog = con.execute("SELECT * FROM product_catalog").df()
    orders = con.execute("SELECT * FROM historical_orders").df()
    con.close()
    bench = pd.read_parquet(GOLD_DIR / "scalability_benchmark.parquet")
    return posts, comments, catalog, orders, bench

posts_df, comments_df, catalog_df, orders_df, bench_df = load_tables()

st.title("UpMark \u2014 Social Audience, Campaign & Order Risk Intelligence")
st.caption("Multilingual (English / Roman Urdu / Urdu) Demand Sensing, Behavioral Clustering & Scale-Adaptive OLAP Engine")

# Sidebar Global Filters
st.sidebar.header("UpMark Campaign Filters")
selected_brands = st.sidebar.multiselect("Select Brands", options=sorted(posts_df["brand"].unique()), default=sorted(posts_df["brand"].unique()))
selected_platforms = st.sidebar.multiselect("Select Platforms", options=sorted(posts_df["platform"].unique()), default=sorted(posts_df["platform"].unique()))
selected_campaigns = st.sidebar.multiselect("Select Campaigns", options=sorted(posts_df["campaign"].unique()), default=sorted(posts_df["campaign"].unique()))

f_posts = posts_df[
    posts_df["brand"].isin(selected_brands)
    & posts_df["platform"].isin(selected_platforms)
    & posts_df["campaign"].isin(selected_campaigns)
]
f_comments = comments_df[comments_df["post_id"].isin(f_posts["post_id"])]

# Top KPI Row
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Analyzed Posts", f"{len(f_posts):,}")
k2.metric("Multilingual Comments", f"{len(f_comments):,}")
k3.metric("Avg Attention Quality (AQI)", f"{f_posts['aqi_score'].mean():.1f} / 100")
k4.metric("Flagged Anomaly Posts", f"{int(f_posts['is_anomaly'].sum())}")
k5.metric("Est. COD Cancellation Risk", f"{f_posts['projected_cod_cancel_risk'].mean() * 100:.1f}%")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Content Attention & AQI",
    "2. Roman Urdu NLP & Audience Clusters",
    "3. Anomaly Radar & COD Order Risk",
    "4. UpMark Action Playbook",
    "5. Big Data Speed & Scale Lab"
])

with tab1:
    st.subheader("Which Content Attracts High-Intent Attention vs. Vanity Reach?")
    col_a, col_b = st.columns(2)
    with col_a:
        fmt_summary = f_posts.groupby("content_type", as_index=False).agg(
            avg_views=("views", "mean"),
            avg_likes=("likes", "mean"),
            avg_aqi=("aqi_score", "mean"),
            purchase_inquiry_share=("Price Inquiry", "mean")
        ).sort_values("avg_aqi", ascending=False)
        fig1 = px.bar(
            fmt_summary, x="content_type", y=["avg_aqi", "avg_likes"],
            barmode="group", title="Attention Quality Index (AQI) vs. Raw Likes by Format"
        )
        st.plotly_chart(fig1, use_container_width=True)
    with col_b:
        plat_summary = f_posts.groupby(["platform", "content_type"], as_index=False)["aqi_score"].mean()
        fig2 = px.bar(
            plat_summary, x="content_type", y="aqi_score", color="platform",
            barmode="group", title="Intent-Weighted AQI Across Platforms (Instagram vs TikTok vs FB)"
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(fmt_summary.round(2), use_container_width=True)

with tab2:
    st.subheader("What Does the Audience Want? (Trilingual NLP & Behavioral Personas)")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        sample_c = f_comments.sample(min(2500, len(f_comments)), random_state=42)
        fig_umap = px.scatter(
            sample_c, x="umap_x", y="umap_y", color="predicted_topic",
            hover_data=["comment_text", "comment_text_clean", "language_detected", "predicted_intent"],
            title="2D Semantic Projection of Multilingual Comments (Roman Urdu + English + Urdu)"
        )
        st.plotly_chart(fig_umap, use_container_width=True)
    with c2:
        persona_counts = f_posts["audience_persona"].value_counts().reset_index()
        persona_counts.columns = ["Audience Persona", "Post Count"]
        fig_pie = px.pie(persona_counts, names="Audience Persona", values="Post Count", hole=0.4, title="Discovered Audience Segments")
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("#### Sample Roman Urdu Phonetic Normalization & Intent Tagging")
    st.dataframe(
        f_comments[["post_id", "brand", "language_detected", "comment_text", "comment_text_clean", "predicted_topic", "predicted_intent", "comment_likes_count"]].head(20),
        use_container_width=True
    )

with tab3:
    st.subheader("Unusual Campaign Patterns (Isolation Forest) & COD Order Cancellation Bridge")
    anomalies = f_posts[f_posts["is_anomaly"]].sort_values("anomaly_score", ascending=False)
    col_x, col_y = st.columns(2)
    with col_x:
        fig_anom = px.scatter(
            f_posts, x="likes_per_day", y="Delivery & Order Delay",
            color="anomaly_type", size="comments_count",
            hover_data=["post_id", "brand", "campaign", "content_type", "aqi_score"],
            title="Contextual Outlier Radar: Engagement Velocity vs. Delivery Complaint Surge"
        )
        st.plotly_chart(fig_anom, use_container_width=True)
    with col_y:
        risk_by_cat = f_posts.groupby(["product_category", "campaign"], as_index=False).agg(
            delivery_complaint_rate=("Delivery & Order Delay", "mean"),
            price_complaint_rate=("Price Complaint", "mean"),
            projected_cod_cancel_risk=("projected_cod_cancel_risk", "mean")
        )
        fig_risk = px.bar(
            risk_by_cat, x="product_category", y="projected_cod_cancel_risk", color="campaign",
            barmode="group", title="Projected Cash-on-Delivery (COD) Cancellation Risk by Category & Campaign"
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    st.markdown("#### Flagged Anomalous Posts Requiring Immediate Investigation")
    st.dataframe(
        anomalies[["post_id", "brand", "platform", "campaign", "content_type", "anomaly_type", "comments_count", "aqi_score", "projected_cod_cancel_risk", "post_url"]].head(15),
        use_container_width=True
    )

with tab4:
    st.subheader("Evidence-Grounded Prescriptive Marketing & CX Action Playbook")
    no_price_posts = f_posts[~f_posts["has_price_in_caption"]]
    price_inq_no_caption = no_price_posts["Price Inquiry"].mean() * 100 if len(no_price_posts) else 0
    price_inq_with_caption = f_posts[f_posts["has_price_in_caption"]]["Price Inquiry"].mean() * 100 if len(f_posts[f_posts["has_price_in_caption"]]) else 0
    top_format = f_posts.groupby("content_type")["aqi_score"].mean().idxmax()
    delivery_crisis_posts = anomalies[anomalies["anomaly_type"].str.contains("Delivery", na=False)]

    st.info(
        f"**1. Content Publishing Strategy:** Prioritize **{top_format}** across active campaigns\u2014it achieves the highest Intent-Weighted Attention Quality Score ({f_posts.groupby('content_type')['aqi_score'].mean().max():.1f}/100) with strong purchase-inquiry conversion."
    )
    st.warning(
        f"**2. Caption & Pricing Optimization:** Posts missing PKR prices in captions trigger **{price_inq_no_caption:.1f}%** 'Price please?' friction vs. **{price_inq_with_caption:.1f}%** when prices are stated upfront. Mandate PKR prices + Karachi/Lahore stock status in all Reel captions."
    )
    st.error(
        f"**3. Urgent Fulfillment & COD Triage:** **{len(delivery_crisis_posts)} active campaign posts** (e.g., `{', '.join(delivery_crisis_posts['post_id'].head(4).tolist())}`) exhibit severe delivery complaint spikes (>35% of comments), pushing projected COD cancellation risk to **{delivery_crisis_posts['projected_cod_cancel_risk'].mean()*100 if len(delivery_crisis_posts) else 32:.1f}%**. Pause ad spend on these posts and dispatch WhatsApp tracking updates immediately."
    )
    st.success(
        "**4. Promotional A/B Test Design:** For segments clustered under *Price-Sensitive & Discount-Hunter*, replace blanket 20% markdowns on >PKR 8,500 Luxury Lawn with a *'Free Express Delivery + PKR 1,000 EasyPaisa/Card Pre-Payment Voucher'* to simultaneously overcome price resistance and cut COD return-to-origin (RTO) losses."
    )

with tab5:
    st.subheader("Scale-Adaptive Data Engineering Benchmark: CSV + Pandas vs. Parquet + DuckDB")
    b1, b2 = st.columns(2)
    with b1:
        fig_t = px.line(
            bench_df, x="rows_processed", y="execution_time_ms", color="engine", markers=True,
            title="Execution Latency (ms) Across Dataset Scale (Lower is Better)"
        )
        st.plotly_chart(fig_t, use_container_width=True)
    with b2:
        fig_m = px.bar(
            bench_df, x="rows_processed", y="peak_ram_mb", color="engine", barmode="group",
            title="Peak Memory Footprint (MB) Across Dataset Scale"
        )
        st.plotly_chart(fig_m, use_container_width=True)
    st.dataframe(bench_df, use_container_width=True)
