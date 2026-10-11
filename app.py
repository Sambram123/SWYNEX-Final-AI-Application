"""
app.py
------
Task 4 – Final AI Application: Interactive Contradiction Detection Dashboard.

Features:
  - Tab 1: Dataset & Product Explorer (Explore 7,109 reviews, filter by product, analyze sentiments & ratings)
  - Tab 2: Batch Contradiction Detector (Run detection on selected products or full dataset, adjust thresholds, rank pairs, export results)
  - Tab 3: Interactive Pair Analyzer (Test any two custom reviews or pick quick examples from the evaluation suite)
  - Tab 4: Evaluation & System Performance (View precision, recall, F1, accuracy, and interactive confusion matrix)
  - Tab 5: Model & Methodology Info (Architecture, MiniLM embedding details, heuristic explanation, limitations)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project modules can be imported
from src.load_data import load_reviews
from src.preprocess import preprocess, rating_to_sentiment
from src.similarity import get_model, encode_texts, cosine_similarity_pair
from src.explainer import (
    compute_confidence,
    generate_explanation,
    enrich_pair,
    rank_pairs,
    validate_pair,
    validate_review,
)
from src.contradiction import detect_contradictions
from src.evaluate import EVAL_DATASET, EXAMPLE_INDICES, _compute_metrics

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Product Review Contradiction Detector",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for polished, modern interface
st.markdown(
    """
    <style>
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.12);
        padding: 14px 18px;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    /* Pair Card Styling */
    .review-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 16px;
    }
    .badge-pos {
        background-color: #10b981;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .badge-neg {
        background-color: #ef4444;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .badge-neu {
        background-color: #6b7280;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .badge-conf {
        background-color: #6366f1;
        color: white;
        padding: 3px 10px;
        border-radius: 14px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Cached Data & Model Loaders
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Sentence-Transformers Model (paraphrase-MiniLM-L3-v2)...")
def load_embedding_model():
    """Load and cache the SentenceTransformer model in memory."""
    return get_model()


@st.cache_data(show_spinner="Loading and preprocessing reviews dataset...")
def get_dataset() -> pd.DataFrame:
    """Load local Parquet file and run preprocessing once."""
    df_raw = load_reviews()
    df_clean = preprocess(df_raw)
    return df_clean


# -----------------------------------------------------------------------------
# Main Application Header
# -----------------------------------------------------------------------------
def render_header():
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("⚖️ AI Product Review Contradiction Detector")
        st.caption(
            "Detect, explain, and rank conflicting customer reviews using "
            "semantic embeddings (**paraphrase-MiniLM-L3-v2**) and sentiment polarity."
        )
    with col2:
        st.info("💡 **Local & Lightweight**\n\nNo external APIs, vector DBs, or GPUs required.")


# -----------------------------------------------------------------------------
# Tab 1: Dataset & Product Explorer
# -----------------------------------------------------------------------------
def render_tab_dataset(df: pd.DataFrame):
    st.header("📊 Dataset & Product Explorer")
    st.write(
        "Explore the Amazon Reviews 2023 Electronics dataset containing **7,109** verified reviews."
    )

    # Top Level Metrics
    total_reviews = len(df)
    unique_products = df["parent_asin"].nunique()
    pos_count = (df["sentiment"] == "Positive").sum()
    neg_count = (df["sentiment"] == "Negative").sum()
    neu_count = (df["sentiment"] == "Neutral").sum()

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Reviews", f"{total_reviews:,}")
    m2.metric("Unique Products", f"{unique_products:,}")
    m3.metric("Positive Reviews", f"{pos_count:,}", f"{(pos_count/total_reviews)*100:.1f}%")
    m4.metric("Negative Reviews", f"{neg_count:,}", f"{(neg_count/total_reviews)*100:.1f}%")
    m5.metric("Neutral Reviews", f"{neu_count:,}", f"{(neu_count/total_reviews)*100:.1f}%")

    st.markdown("---")

    # Visualizations Row
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.subheader("Sentiment Distribution")
        fig_sent = px.pie(
            names=["Positive", "Negative", "Neutral"],
            values=[pos_count, neg_count, neu_count],
            color=["Positive", "Negative", "Neutral"],
            color_discrete_map={
                "Positive": "#10b981",
                "Negative": "#ef4444",
                "Neutral": "#9ca3af",
            },
            hole=0.45,
        )
        fig_sent.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300)
        st.plotly_chart(fig_sent, use_container_width=True)

    with col_chart2:
        st.subheader("Star Rating Distribution (1–5)")
        rating_counts = df["rating"].value_counts().sort_index().reset_index()
        rating_counts.columns = ["Rating", "Count"]
        fig_rating = px.bar(
            rating_counts,
            x="Rating",
            y="Count",
            color="Rating",
            color_continuous_scale="Teal",
        )
        fig_rating.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300)
        st.plotly_chart(fig_rating, use_container_width=True)

    # Product-level breakdown
    st.subheader("🔍 Search & Inspect Product Reviews")
    
    # Calculate contradiction potential per product
    product_stats = (
        df.groupby(["parent_asin", "product_title"])
        .agg(
            total_reviews=("rating", "count"),
            pos_reviews=("sentiment", lambda s: (s == "Positive").sum()),
            neg_reviews=("sentiment", lambda s: (s == "Negative").sum()),
            avg_rating=("rating", "mean"),
        )
        .reset_index()
    )
    product_stats["conflict_potential"] = (
        (product_stats["pos_reviews"] > 0) & (product_stats["neg_reviews"] > 0)
    )

    # Filter options
    f_col1, f_col2 = st.columns([2, 1])
    with f_col1:
        search_query = st.text_input("Filter products by title or ASIN:", placeholder="e.g. Fire TV, Earbuds, B075X8471B")
    with f_col2:
        only_conflicts = st.checkbox("Show only products with both Positive & Negative reviews", value=True)

    filtered_products = product_stats.copy()
    if search_query.strip():
        q = search_query.strip().lower()
        filtered_products = filtered_products[
            filtered_products["product_title"].str.lower().str.contains(q, na=False)
            | filtered_products["parent_asin"].str.lower().str.contains(q, na=False)
        ]
    if only_conflicts:
        filtered_products = filtered_products[filtered_products["conflict_potential"]]

    filtered_products = filtered_products.sort_values("total_reviews", ascending=False)

    st.write(f"Showing **{len(filtered_products):,}** matching products:")
    st.dataframe(
        filtered_products[["parent_asin", "product_title", "total_reviews", "pos_reviews", "neg_reviews", "avg_rating"]].rename(
            columns={
                "parent_asin": "ASIN",
                "product_title": "Product Title",
                "total_reviews": "Total Reviews",
                "pos_reviews": "Positive (4-5★)",
                "neg_reviews": "Negative (1-2★)",
                "avg_rating": "Avg Rating",
            }
        ),
        use_container_width=True,
        height=260,
    )


# -----------------------------------------------------------------------------
# Tab 2: Batch Contradiction Detector
# -----------------------------------------------------------------------------
def render_tab_batch_detector(df: pd.DataFrame):
    st.header("⚡ Batch Contradiction Detector")
    st.write(
        "Scan reviews across products to surface the strongest conflicting opinions."
    )

    # Sidebar / Parameter settings
    st.sidebar.subheader("⚙️ Detector Parameters")
    similarity_thresh = st.sidebar.slider(
        "Semantic Similarity Threshold",
        min_value=0.30,
        max_value=0.90,
        value=0.55,
        step=0.05,
        help="Minimum cosine similarity required between two reviews to consider them discussing the same aspect.",
    )
    max_pairs_per_prod = st.sidebar.number_input(
        "Max Pairs Per Product",
        min_value=1,
        max_value=20,
        value=5,
        help="Prevents a single product with many reviews from dominating the results.",
    )
    top_n_results = st.sidebar.number_input(
        "Top N Pairs to Display",
        min_value=5,
        max_value=100,
        value=15,
        step=5,
        help="Total number of highest-confidence contradictory pairs to return.",
    )

    # Product Selection Mode
    c1, c2 = st.columns([2, 1])
    with c1:
        # Prepopulate with top products that have both positive and negative reviews
        conflict_prods = (
            df.groupby("parent_asin")
            .filter(lambda g: ((g["sentiment"] == "Positive").sum() > 0) and ((g["sentiment"] == "Negative").sum() > 0))
        )
        top_asins = conflict_prods["parent_asin"].value_counts().head(20).index.tolist()
        
        asin_label_map = {}
        for a in top_asins:
            title = df[df["parent_asin"] == a]["product_title"].iloc[0]
            count = len(df[df["parent_asin"] == a])
            asin_label_map[a] = f"{title[:65]}... ({a}, {count} reviews)"

        options = ["All Products (Full Scan)"] + [asin_label_map[a] for a in top_asins]
        selected_option = st.selectbox(
            "Select Scope:",
            options=options,
            index=1,  # Default to first top product
            help="Select a specific high-conflict product for fast scanning or run across the dataset.",
        )

    with c2:
        st.write("")
        st.write("")
        run_btn = st.button("🚀 Run Contradiction Detection", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("Analyzing semantic embeddings and finding contradictions..."):
            start_t = time.time()
            if selected_option == "All Products (Full Scan)":
                target_df = df
            else:
                # Extract ASIN
                selected_asin = [k for k, v in asin_label_map.items() if v == selected_option][0]
                target_df = df[df["parent_asin"] == selected_asin]

            results = detect_contradictions(
                target_df,
                similarity_threshold=similarity_thresh,
                max_pairs_per_product=max_pairs_per_prod,
                top_n=top_n_results,
            )
            elapsed = time.time() - start_t

        st.success(f"Detection completed in **{elapsed:.2f}s**! Found **{len(results)}** contradictory pairs.")

        if not results:
            st.warning(
                f"No contradictory pairs met the criteria with similarity threshold >= {similarity_thresh:.2f}. "
                "Try lowering the threshold slightly in the sidebar (e.g. 0.45 - 0.50)."
            )
            return

        # Summary Metrics
        avg_conf = np.mean([p["confidence"] for p in results])
        high_tier_count = sum(1 for p in results if p.get("confidence_label") in ("Very High", "High"))

        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Contradictions Found", len(results))
        sc2.metric("Avg Confidence", f"{avg_conf*100:.1f}%")
        sc3.metric("High / Very High Confidence", high_tier_count)

        # Download button
        st.markdown("---")
        json_str = json.dumps(results, indent=2, ensure_ascii=False)
        st.download_button(
            label="📥 Export Results as JSON",
            data=json_str,
            file_name="contradiction_results.json",
            mime="application/json",
        )

        st.subheader("📋 Ranked Contradiction Pairs")

        for idx, pair in enumerate(results, start=1):
            with st.expander(
                f"Pair #{idx} | Confidence: {pair['confidence']*100:.1f}% ({pair['confidence_label']}) | "
                f"Similarity: {pair['similarity_score']:.2f} | {pair['product_title'][:60]}...",
                expanded=(idx <= 3),
            ):
                st.markdown(f"**Product**: {pair['product_title']}")
                st.markdown(f"**ASIN**: `{pair['parent_asin']}`")

                # Explanation callout
                st.info(f"💡 **AI Explanation**: {pair['explanation']}")

                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    badge_cls = "badge-pos" if pair["sentiment_1"] == "Positive" else "badge-neg"
                    st.markdown(f"<span class='{badge_cls}'>{pair['sentiment_1']} Review</span>", unsafe_allow_html=True)
                    st.markdown(f"> *\"{pair['review_1']}\"*")

                with col_r2:
                    badge_cls = "badge-pos" if pair["sentiment_2"] == "Positive" else "badge-neg"
                    st.markdown(f"<span class='{badge_cls}'>{pair['sentiment_2']} Review</span>", unsafe_allow_html=True)
                    st.markdown(f"> *\"{pair['review_2']}\"*")


# -----------------------------------------------------------------------------
# Tab 3: Interactive Pair Analyzer
# -----------------------------------------------------------------------------
def render_tab_interactive_analyzer():
    st.header("🔬 Interactive Review Pair Analyzer")
    st.write(
        "Test custom pairs of reviews or select preset examples from the evaluation suite to see "
        "how the model computes semantic similarity, extracts sentiment polarity, and explains the contradiction."
    )

    preset_options = [
        "Custom Input (Enter your own)",
        "Example 1: Battery Life Contradiction (Positive 5★ vs Negative 1★)",
        "Example 2: Sound Quality Contradiction (Positive 5★ vs Negative 1★)",
        "Example 3: Build Quality Contradiction (Positive 4★ vs Negative 2★)",
        "Example 4: Bluetooth Connectivity (Positive 5★ vs Negative 1★)",
        "Example 5: Agreement / Non-Contradiction (Both Positive 5★)",
        "Example 6: Different Products / Aspects (No topical overlap)",
    ]

    preset_choice = st.selectbox("Load Sample Review Pair:", preset_options)

    # Default values based on preset
    default_r1, default_rate1 = "", 5.0
    default_r2, default_rate2 = "", 1.0

    if preset_choice == "Example 1: Battery Life Contradiction (Positive 5★ vs Negative 1★)":
        default_r1 = "The battery easily lasts two full days even with heavy use. Outstanding performance."
        default_rate1 = 5.0
        default_r2 = "The battery barely lasts five hours. Absolutely terrible battery life."
        default_rate2 = 1.0
    elif preset_choice == "Example 2: Sound Quality Contradiction (Positive 5★ vs Negative 1★)":
        default_r1 = "Sound quality is exceptional. Rich bass and crystal clear highs. Best headphones I have ever owned."
        default_rate1 = 5.0
        default_r2 = "Horrible sound quality. Tinny and distorted at any volume. Complete waste of money."
        default_rate2 = 1.0
    elif preset_choice == "Example 3: Build Quality Contradiction (Positive 4★ vs Negative 2★)":
        default_r1 = "Build quality is superb. Very sturdy and feels premium."
        default_rate1 = 4.0
        default_r2 = "Cheap plastic build. Fell apart within a week of light use."
        default_rate2 = 2.0
    elif preset_choice == "Example 4: Bluetooth Connectivity (Positive 5★ vs Negative 1★)":
        default_r1 = "Connects to Bluetooth instantly every time. Rock solid connection."
        default_rate1 = 5.0
        default_r2 = "Bluetooth drops every few minutes. Constant disconnections ruin the experience."
        default_rate2 = 1.0
    elif preset_choice == "Example 5: Agreement / Non-Contradiction (Both Positive 5★)":
        default_r1 = "The display is incredibly bright and sharp. Colors look totally vibrant."
        default_rate1 = 5.0
        default_r2 = "Screen is crystal clear and vivid. Gorgeous display for watching videos."
        default_rate2 = 5.0
    elif preset_choice == "Example 6: Different Products / Aspects (No topical overlap)":
        default_r1 = "Fast shipping and arrived in pristine packaging."
        default_rate1 = 5.0
        default_r2 = "The battery drains way too quickly."
        default_rate2 = 1.0

    col_in1, col_in2 = st.columns(2)
    with col_in1:
        st.subheader("Review #1")
        text_1 = st.text_area("Review Text 1:", value=default_r1, height=130, placeholder="Enter first review...")
        rating_1 = st.slider("Star Rating 1:", min_value=1.0, max_value=5.0, value=default_rate1, step=1.0)
        s1 = rating_to_sentiment(rating_1)
        st.caption(f"Derived Sentiment: **{s1}**")

    with col_in2:
        st.subheader("Review #2")
        text_2 = st.text_area("Review Text 2:", value=default_r2, height=130, placeholder="Enter second review...")
        rating_2 = st.slider("Star Rating 2:", min_value=1.0, max_value=5.0, value=default_rate2, step=1.0)
        s2 = rating_to_sentiment(rating_2)
        st.caption(f"Derived Sentiment: **{s2}**")

    thresh = st.slider("Analysis Similarity Threshold", min_value=0.30, max_value=0.90, value=0.60, step=0.05)

    if st.button("🔍 Analyze Review Pair", type="primary", use_container_width=True):
        # 1. Validation
        valid, msg = validate_pair(text_1, rating_1, text_2, rating_2)
        if not valid:
            st.error(f"❌ Input Validation Failed: {msg}")
            return

        with st.spinner("Computing MiniLM sentence embeddings..."):
            # Compute embeddings
            embeddings = encode_texts([text_1, text_2], show_progress=False)
            similarity = cosine_similarity_pair(embeddings[0], embeddings[1])
            confidence, verdict = compute_confidence(s1, s2, similarity, threshold=thresh)
            explanation = generate_explanation(verdict, s1, s2, similarity, confidence)

        st.markdown("---")
        st.subheader("Analysis Results")

        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
        verdict_str = "CONTRADICTION" if verdict == "contradiction" else "NON-CONTRADICTION"
        color_verdict = "normal" if verdict == "contradiction" else "off"

        res_col1.metric("Verdict", verdict_str)
        res_col2.metric("Cosine Similarity", f"{similarity:.4f}", f"{similarity*100:.1f}%")
        res_col3.metric("Confidence Score", f"{confidence*100:.1f}%" if confidence > 0 else "0.0%")
        tier_label = (
            "Very High" if confidence >= 0.90 else
            "High" if confidence >= 0.75 else
            "Moderate" if confidence >= 0.60 else
            "Low" if confidence > 0 else "N/A"
        )
        res_col4.metric("Confidence Tier", tier_label)

        if verdict == "contradiction":
            st.success(f"✅ **Contradiction Detected**: {explanation}")
        else:
            st.warning(f"ℹ️ **No Contradiction**: {explanation}")

        # Visual similarity bar
        st.markdown(f"**Semantic Topic Overlap**: `{similarity*100:.1f}%`")
        st.progress(max(0.0, min(1.0, float(similarity))))


# -----------------------------------------------------------------------------
# Tab 4: Evaluation & Performance
# -----------------------------------------------------------------------------
def render_tab_evaluation():
    st.header("📈 Evaluation & System Performance")
    st.write(
        "Empirical benchmark results on the **25-item manually labeled test dataset** "
        "designed in Task 3 to evaluate precision, recall, accuracy, and F1-score."
    )

    eval_thresh = st.slider("Evaluation Threshold", min_value=0.40, max_value=0.80, value=0.60, step=0.05)

    with st.spinner("Running evaluation suite over labeled pairs..."):
        # Run evaluation dynamically for selected threshold
        all_texts = []
        for item in EVAL_DATASET:
            all_texts.append(item["review_1"])
            all_texts.append(item["review_2"])
        embeddings = encode_texts(all_texts, show_progress=False)

        y_true, y_pred, records = [], [], []
        for idx, item in enumerate(EVAL_DATASET):
            r1, r1_rate = item["review_1"], item["rating_1"]
            r2, r2_rate = item["review_2"], item["rating_2"]
            expected = item["expected_label"]

            ok, _ = validate_pair(r1, r1_rate, r2, r2_rate)
            if not ok:
                pred = 0
                conf = 0.0
                v = "invalid"
            else:
                s1 = rating_to_sentiment(float(r1_rate))
                s2 = rating_to_sentiment(float(r2_rate))
                sim = cosine_similarity_pair(embeddings[idx * 2], embeddings[idx * 2 + 1])
                conf, v = compute_confidence(s1, s2, sim, threshold=eval_thresh)
                pred = 1 if v == "contradiction" else 0

            y_true.append(expected)
            y_pred.append(pred)
            records.append({
                "Index": idx + 1,
                "Note": item.get("note", ""),
                "Expected": "Contradiction" if expected == 1 else "Non-Contradiction",
                "Predicted": "Contradiction" if pred == 1 else "Non-Contradiction",
                "Status": "✅ Correct" if pred == expected else "❌ Incorrect",
                "Confidence": f"{conf*100:.0f}%" if conf > 0 else "--",
                "Review 1": r1[:65] + "...",
                "Review 2": r2[:65] + "...",
            })

        metrics = _compute_metrics(y_true, y_pred)

    # Metrics Display
    ec1, ec2, ec3, ec4 = st.columns(4)
    ec1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
    ec2.metric("Precision", f"{metrics['precision']*100:.1f}%")
    ec3.metric("Recall", f"{metrics['recall']*100:.1f}%")
    ec4.metric("F1-Score", f"{metrics['f1']*100:.1f}%")

    st.markdown("---")

    # Confusion Matrix Visualization
    cm_col1, cm_col2 = st.columns([1, 1])
    with cm_col1:
        st.subheader("Confusion Matrix")
        z = [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]]
        x = ["Predicted Non-Contradiction", "Predicted Contradiction"]
        y = ["Actual Non-Contradiction", "Actual Contradiction"]

        fig_cm = px.imshow(
            z,
            x=x,
            y=y,
            text_auto=True,
            color_continuous_scale="Blues",
            aspect="auto",
        )
        fig_cm.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=300)
        st.plotly_chart(fig_cm, use_container_width=True)

    with cm_col2:
        st.subheader("Confusion Matrix Counts")
        st.write(f"- **True Positives (TP)**: {metrics['tp']} (correctly identified contradictions)")
        st.write(f"- **True Negatives (TN)**: {metrics['tn']} (correctly identified non-contradictions)")
        st.write(f"- **False Positives (FP)**: {metrics['fp']} (incorrectly flagged contradictions)")
        st.write(f"- **False Negatives (FN)**: {metrics['fn']} (missed contradictions)")

    st.subheader("Dataset Predictions Table")
    st.dataframe(pd.DataFrame(records), use_container_width=True, height=350)


# -----------------------------------------------------------------------------
# Tab 5: Model & Methodology Info
# -----------------------------------------------------------------------------
def render_tab_info():
    st.header("ℹ️ Architecture & Methodology")

    st.markdown(
        """
        ### 1. Problem Formulation
        Product review contradiction detection identifies pairs of customer reviews for the **same product**
        that express diametrically opposed opinions on the **same specific feature or aspect**.

        ### 2. Detection Pipeline
        ```
        Raw Reviews ─► Preprocess & Clean ─► Sentiment Polarity Assignment
                                                        │
        paraphrase-MiniLM-L3-v2 Model ──────────────► Cosine Similarity Matrix
                                                        │
        Rule-Based Contradiction Logic ◄────────────────┘
              │  (Opposite Sentiments + Similarity >= Threshold)
              ▼
        Intelligent Explanation & Confidence Scoring ─► Ranked Contradiction Pairs
        ```

        ### 3. Model Architecture: `paraphrase-MiniLM-L3-v2`
        - **Model Size**: ~17 MB (extremely lightweight)
        - **Embedding Dimension**: 384
        - **Parameters**: ~15M
        - **L2 Normalization**: Pre-normalized so cosine similarity is computed directly via vector dot product
        - **Local Execution**: Runs 100% on CPU without requiring GPU or remote API calls

        ### 4. Sentiment Polarity Mapping
        - **Ratings 4.0 – 5.0**: `Positive`
        - **Rating 3.0**: `Neutral`
        - **Ratings 1.0 – 2.0**: `Negative`
        - *Contradictions require strict `Positive` ↔ `Negative` polarity matching.*

        ### 5. Confidence Score Heuristic
        The confidence score is defined as:
        $$\\text{Confidence} = \\text{Cosine Similarity} \\quad (\\text{when } \\text{Polarity matches and Similarity} \\ge \\tau)$$
        - **Very High (90–100%)**: Almost identical topic wording, polar opposite sentiment.
        - **High (75–89%)**: Strong aspect overlap with clear disagreement.
        - **Moderate (60–74%)**: Related general topic, conflicting satisfaction.
        - **Low (< 60%)**: Weak similarity; insufficient confidence.

        ### 6. Limitations & Scope
        - **Rule-Based Prototype**: Does not perform external fact-verification or detect sarcasm.
        - **Threshold Sensitivity**: Highly specific product aspects may require lower thresholds (~0.50).
        """
    )


# -----------------------------------------------------------------------------
# Main Application Entrypoint
# -----------------------------------------------------------------------------
def main():
    render_header()

    # Preload Model & Data
    with st.spinner("Initializing AI model and reading dataset..."):
        load_embedding_model()
        df = get_dataset()

    # Create Navigation Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 Dataset Explorer",
            "⚡ Batch Detector",
            "🔬 Interactive Pair Analyzer",
            "📈 Evaluation & Metrics",
            "ℹ️ Architecture & Info",
        ]
    )

    with tab1:
        render_tab_dataset(df)

    with tab2:
        render_tab_batch_detector(df)

    with tab3:
        render_tab_interactive_analyzer()

    with tab4:
        render_tab_evaluation()

    with tab5:
        render_tab_info()


if __name__ == "__main__":
    main()
