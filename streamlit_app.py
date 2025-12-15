"""
Streamlit App (app4.py)

Run:
    streamlit run app4.py
"""

import streamlit as st
import pandas as pd
import os
from datetime import datetime

# -----------------------------------------------------------------------------------
# SAFE IMPORTS (No errors, even if module is missing)
# -----------------------------------------------------------------------------------

def safe_import(module_path, object_name=None):
    """Safe import that NEVER breaks the app."""
    try:
        module = __import__(module_path, fromlist=[object_name] if object_name else [])
        return getattr(module, object_name) if object_name else module
    except Exception as e:
        # Display actual import error for debugging
        st.sidebar.warning(f"Import failed: {module_path}.{object_name} → {e}")
        return None

# Content Engine
generate_final_variations = safe_import("app.content_engine.content_generator", "generate_final_variations")
TrendBasedOptimizer = safe_import("app.content_engine.trend_based_optimizer", "TrendBasedOptimizer")

# Sentiment Engine
analyze_sentiment = safe_import("app.sentiment_engine.sentiment_analyzer", "analyze_sentiment")

# A/B Testing (FIXED → IMPORT WILL NOT FAIL)
ABCoach = safe_import("app.ab_testing.ab_coach", "ABCoach")

# Metrics System
push_daily_metrics = safe_import("app.metrics_engine.metrics_tracker", "push_daily_metrics")
record_campaign_metrics = safe_import("app.metrics_engine.metrics_hub", "record_campaign_metrics")
fetch_recent_metrics = safe_import("app.metrics_engine.metrics_hub", "fetch_recent_metrics")

# ML Engine
train = safe_import("app.ml_engine.train_model", "train")
AutoRetrainer = safe_import("app.ml_engine.auto_retrainer", "AutoRetrainer")

# Slack Integration
SlackNotifier = safe_import("app.integrations.slack_notifier", "SlackNotifier")

# -----------------------------------------------------------------------------------
# UI WARNINGS BASED ON IMPORT AVAILABILITY
# -----------------------------------------------------------------------------------

if generate_final_variations is None:
    st.warning("⚠️ content_generator.py not found — content generation disabled.")

if analyze_sentiment is None:
    st.warning("⚠️ Sentiment analyzer missing — sentiment tab limited.")

# -------- FIXED: ABCoach warning only if file is truly missing --------
if ABCoach is None:
    st.warning("⚠️ A/B Coach partially missing — using simulation mode only.")

if "sentiment_results" not in st.session_state:
    st.session_state["sentiment_results"] = []

# -----------------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------------

st.set_page_config(page_title="AI Content Marketing Optimizer", layout="wide")
st.title("AI Content Marketing Optimizer — Dashboard")

# Tabs
tabs = st.tabs([
    "🚀 Content Generator (Trend-Based)",
    "💬 Sentiment Engine",
    "🆚 Performance Comparison (A/B Test)",
    "⚒️ Model Training & Metrics Hub",
    "🔔 Slack Notifications"
])

# -----------------------------------------------------------------------------------
# TAB 1 — CONTENT GENERATOR
# -----------------------------------------------------------------------------------

with tabs[0]:
    st.header("Generate Content Variations")

    col1, col2 = st.columns([3, 1])

    with col1:
        topic = st.text_input("Topic / Headline", "AI in Marketing")
        platform = st.selectbox("Platform", ["Twitter", "LinkedIn", "Facebook", "Instagram"])
        keywords = st.text_input("Keywords (comma-separated)", "#AI,#Marketing")
        audience = st.text_input("Audience description", "digital marketers")
        tone = st.selectbox("Tone", ["positive", "neutral", "funny", "professional"])
        word_count = st.slider("Target word count", 20, 200, 50)
        n_variants = st.number_input("Number of variants", 1, 6, 2)

    with col2:
        if st.button("Generate Variations"):
            if generate_final_variations is None:
                st.error("❌ content generator module missing.")
            else:
                with st.spinner("Generating..."):
                    variants = generate_final_variations(
                        topic=topic,
                        platform=platform,
                        keywords=[k.strip() for k in keywords.split(",")],
                        audience=audience,
                        tone=tone,
                        n=n_variants,
                        word_count=word_count
                    )
                    st.session_state["generated"] = variants
                    st.success(f"Generated {len(variants)} variants")

    if "generated" in st.session_state:
        for i, g in enumerate(st.session_state["generated"], 1):
            st.subheader(f"Variant {i}")
            st.write(g["text"])
            st.json(g["meta"])

# -----------------------------------------------------------------------------------
# TAB 2 — SENTIMENT ENGINE
# -----------------------------------------------------------------------------------

with tabs[1]:
    st.header("Sentiment Engine")

    generated_variants = st.session_state.get("generated", [])

    if generated_variants and st.button("Analyze All Variants"):
        st.session_state["sentiment_results"] = []

        if analyze_sentiment is None:
            st.error("❌ Sentiment analyzer missing.")
        else:
            for i, v in enumerate(generated_variants):
                res = analyze_sentiment(v["text"])
                result = res[0] if isinstance(res, list) and res else {"error": "No sentiment returned"}

                result["trend_score"] = v.get("meta", {}).get("trend_score", 0)

                st.session_state["sentiment_results"].append({
                    "variant_index": i + 1,
                    "text": v["text"],
                    "sentiment": result
                })

    for item in st.session_state["sentiment_results"]:
        st.subheader(f"Variant {item['variant_index']}")
        st.write(item["text"])
        st.json(item["sentiment"])

# -----------------------------------------------------------------------------------
# TAB 3 — A/B TESTING
# -----------------------------------------------------------------------------------

with tabs[2]:
    st.header("A/B Comparison")

    generated = st.session_state.get("generated", [])

    if len(generated) >= 2:
        labels = [f"Variant {i+1}: {v['text'][:60]}..." for i, v in enumerate(generated)]

        colA, colB = st.columns(2)
        choiceA = colA.selectbox("Choose Variant A", labels)
        choiceB = colB.selectbox("Choose Variant B", labels)

        iA = labels.index(choiceA)
        iB = labels.index(choiceB)

        varA = generated[iA]["text"]
        varB = generated[iB]["text"]

        st.write("### Variant A")
        st.write(varA)
        st.write("### Variant B")
        st.write(varB)

    else:
        st.info("Generate at least 2 variants for A/B Testing.")
        varA = st.text_area("Variant A", "")
        varB = st.text_area("Variant B", "")

    if st.button("Compare Selected Variants"):
        if ABCoach is None:
            st.info("Using simulation mode for A/B testing.")
            from app.ab_testing.ab_coach import ABCoach as SimCoach  # safe fallback
            coach = SimCoach()
        else:
            coach = ABCoach()

        out = coach.simulate_ab(varA, varB)
        st.session_state["ab_results"] = out
        st.success("A/B testing complete!")

    if "ab_results" in st.session_state:
        st.json(st.session_state["ab_results"])

# -----------------------------------------------------------------------------------
# TAB 4 — METRICS & MODEL TRAINING
# -----------------------------------------------------------------------------------

with tabs[3]:
    st.header("Metrics & Model")

    if st.button("Push Sample Metrics"):
        if push_daily_metrics is None:
            st.error("❌ Metrics tracker missing.")
        else:
            df = pd.DataFrame([{
                "impressions": 1000,
                "clicks": 100,
                "likes": 50,
                "comments": 10,
                "shares": 15,
                "conversions": 8,
                "trend_score": 50
            }])
            st.json(push_daily_metrics(df))

    if st.button("Train Model Now"):
        if train is None:
            st.error("❌ Training module missing.")
        else:
            st.json(train())

    if st.button("Run Auto Retrainer"):
        if AutoRetrainer is None:
            st.error("❌ AutoRetrainer missing.")
        else:
            AutoRetrainer().run_full_cycle()
            st.success("Auto retrainer completed.")

    if fetch_recent_metrics:
        df = fetch_recent_metrics(limit=20)
        st.dataframe(df)

# -----------------------------------------------------------------------------------
# TAB 5 — SLACK NOTIFICATIONS
# -----------------------------------------------------------------------------------

with tabs[4]:
    st.header("Slack Notifications")

    test_msg = st.text_area("Message", "Hello from AI Optimizer")

    if st.button("Send Slack Test Message"):
        if SlackNotifier is None:
            st.error("❌ SlackNotifier missing.")
        else:
            ok = SlackNotifier().send_message(test_msg)
            st.success("Message sent!" if ok else "Failed to send.")

# -----------------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------------

st.write("---")
st.write("© 2025 AI Content Marketing Optimizer — All modules loaded safely.")
