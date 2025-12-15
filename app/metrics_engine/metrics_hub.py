# metrics_hub.py (FULLY FIXED VERSION)

import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
import pandas as pd

# ---------------------------------------------
# Paths
# ---------------------------------------------
DATA_DIR = "data"
CAMPAIGNS_CSV = os.path.join(DATA_DIR, "campaigns1.csv")
HISTORICAL_CSV = os.path.join(DATA_DIR, "historical_metrics.csv")

os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------------
# Logging
# ---------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

# ---------------------------------------------
# Optional integrations (SAFE IMPORTS)
# ---------------------------------------------
try:
    from app.integrations.sheets_connector import append_row
    SHEETS_AVAILABLE = True
except Exception:
    append_row = None
    SHEETS_AVAILABLE = False
    logger.info("Sheets connector NOT available")

try:
    from app.integrations.social_ingestor import SocialIngestor
    _SOCIAL_INGESTOR_AVAILABLE = True
    _INGESTOR = SocialIngestor()
except Exception as e:
    _SOCIAL_INGESTOR_AVAILABLE = False
    _INGESTOR = None
    logger.info(f"SocialIngestor import failed: {e}")

try:
    from app.integrations.trend_fetcher import TrendFetcher
    _TREND_AVAILABLE = True
    _TRENDER = TrendFetcher()
except Exception as e:
    _TREND_AVAILABLE = False
    _TRENDER = None
    logger.info(f"TrendFetcher import failed: {e}")

# Sentiment engine
try:
    from app.sentiment_engine.sentiment_analyzer import analyze_sentiment, analyze_post_comments
    _SENTIMENT_AVAILABLE = True
except Exception as e:
    analyze_sentiment = None
    analyze_post_comments = None
    _SENTIMENT_AVAILABLE = False
    logger.info(f"Sentiment Analyzer import failed: {e}")

GOOGLE_SHEET_ENABLED = bool(os.getenv("GOOGLE_SHEET_ID")) and SHEETS_AVAILABLE

# ---------------------------------------------
# Initialize local CSV files
# ---------------------------------------------
def _init_file(path: str, columns: List[str]):
    if not os.path.exists(path):
        pd.DataFrame(columns=columns).to_csv(path, index=False)
        logger.info(f"Created: {path}")

_init_file(CAMPAIGNS_CSV, [
    "timestamp", "campaign_id", "variant", "post_id", "platform",
    "impressions", "clicks", "conversions", "ctr", "conv_rate",
    "sentiment", "trend_score", "polarity", "avg_comment_sentiment"
])

_init_file(HISTORICAL_CSV, [
    "timestamp", "campaign_id", "variant",
    "ctr", "sentiment", "polarity", "conversions", "trend_score"
])

# ---------------------------------------------
# Core Writer
# ---------------------------------------------
def record_campaign_metrics(
    campaign_id: str,
    variant: str,
    impressions: int,
    clicks: int,
    conversions: int,
    sentiment_score: float,
    trend_score: float = 0.0,
    platform: str = "unknown",
    post_id: Optional[str] = None,
    avg_comment_sentiment: Optional[float] = None
) -> None:

    timestamp = datetime.utcnow().isoformat()

    ctr = (clicks / impressions) if impressions > 0 else 0.0
    conv_rate = (conversions / clicks) if clicks > 0 else 0.0

    row = {
        "timestamp": timestamp,
        "campaign_id": campaign_id,
        "variant": variant,
        "post_id": post_id or "",
        "platform": platform,
        "impressions": impressions,
        "clicks": clicks,
        "conversions": conversions,
        "ctr": ctr,
        "conv_rate": conv_rate,
        "sentiment": float(sentiment_score),
        "trend_score": float(trend_score),
        "polarity": float(sentiment_score),
        "avg_comment_sentiment": float(avg_comment_sentiment)
        if avg_comment_sentiment is not None else None
    }

    # save to campaigns CSV
    try:
        df = pd.read_csv(CAMPAIGNS_CSV)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(CAMPAIGNS_CSV, index=False)
    except Exception as e:
        logger.error(f"Failed writing campaigns CSV: {e}")

    # save to historical (ML dataset)
    try:
        hrow = {
            "timestamp": timestamp,
            "campaign_id": campaign_id,
            "variant": variant,
            "ctr": ctr,
            "sentiment": float(sentiment_score),
            "polarity": float(sentiment_score),
            "conversions": conversions,
            "trend_score": trend_score
        }
        hdf = pd.read_csv(HISTORICAL_CSV)
        hdf = pd.concat([hdf, pd.DataFrame([hrow])], ignore_index=True)
        hdf.to_csv(HISTORICAL_CSV, index=False)
    except Exception as e:
        logger.error(f"Failed writing historical CSV: {e}")

    # Google Sheets
    if GOOGLE_SHEET_ENABLED:
        try:
            append_row("campaigns", [
                timestamp, campaign_id, variant, post_id or "", platform,
                impressions, clicks, conversions,
                round(ctr, 6), round(conv_rate, 6),
                sentiment_score, trend_score,
                avg_comment_sentiment if avg_comment_sentiment else ""
            ])
        except Exception as e:
            logger.warning(f"Sheets append failed: {e}")

# ---------------------------------------------
# AUTO INGEST FROM POST ID
# ---------------------------------------------
def record_post_metrics_from_id(campaign_id: str, variant: str, post_id: str, platform: str = "twitter"):

    if not _SOCIAL_INGESTOR_AVAILABLE:
        logger.error("SocialIngestor unavailable")
        return

    try:
        data = _INGESTOR.fetch_complete_post_data(str(post_id))
        metrics = data.get("metrics", {})
        text = data.get("text", "")

        impressions = metrics.get("impressions") or 0
        likes = metrics.get("likes") or 0
        shares = metrics.get("shares") or 0
        replies = metrics.get("replies") or 0

        clicks = 0
        conversions = 0

        # sentiment
        sentiment_score = 0.5
        avg_comment = None

        if _SENTIMENT_AVAILABLE and analyze_sentiment:
            s = analyze_sentiment(text)[0]
            sentiment_score = s.get("sentiment_score", 0.5)

            cstats = analyze_post_comments(post_id)
            if isinstance(cstats, dict):
                avg_comment = cstats.get("avg_sentiment", None)

        # trend
        trend_score = 0.0
        if _TREND_AVAILABLE:
            try:
                trend_score = _TRENDER.get_combined_trend_score(text)
            except:
                trend_score = 0.0

        # fallback for impressions
        if not impressions:
            impressions = (likes + shares + replies) * 100

        # SAVE
        record_campaign_metrics(
            campaign_id, variant,
            impressions=int(impressions),
            clicks=clicks,
            conversions=conversions,
            sentiment_score=float(sentiment_score),
            trend_score=float(trend_score),
            platform=platform,
            post_id=post_id,
            avg_comment_sentiment=avg_comment
        )

    except Exception as e:
        logger.error(f"Post ingestion failed: {e}")

# ---------------------------------------------
# Query Utilities
# ---------------------------------------------
def fetch_recent_metrics(limit: int = 50) -> pd.DataFrame:
    try:
        df = pd.read_csv(CAMPAIGNS_CSV)
        return df.tail(limit)
    except:
        return pd.DataFrame()

def fetch_campaign_history(campaign_id: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(CAMPAIGNS_CSV)
        return df[df["campaign_id"] == campaign_id]
    except:
        return pd.DataFrame()

def fetch_variant_performance(campaign_id: str) -> Dict:
    df = fetch_campaign_history(campaign_id)
    if df.empty:
        return {}

    df["ctr"] = pd.to_numeric(df["ctr"], errors="coerce").fillna(0.0)
    df["conv_rate"] = pd.to_numeric(df["conv_rate"], errors="coerce").fillna(0.0)

    df_sorted = df.sort_values(["conv_rate", "ctr"], ascending=False)

    return {
        "best": df_sorted.iloc[0].to_dict(),
        "worst": df_sorted.iloc[-1].to_dict(),
        "all_variants": df_sorted.to_dict(orient="records")
    }

def get_ml_training_data() -> pd.DataFrame:
    try:
        df = pd.read_csv(HISTORICAL_CSV)
        return df.dropna()
    except:
        return pd.DataFrame()

# ---------------------------------------------
# Feature utils
# ---------------------------------------------
def build_feature_vector(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ctr": float(row.get("ctr", 0.0)),
        "sentiment": float(row.get("sentiment", 0.0)),
        "polarity": float(row.get("polarity", 0.0)),
        "trend_score": float(row.get("trend_score", 0.0)),
        "conversions": int(row.get("conversions", 0))
    }

def compute_variant_score(row: Dict[str, Any]) -> float:
    try:
        return round(
            float(row.get("ctr", 0)) * 0.5 +
            float(row.get("sentiment", 0)) * 0.3 +
            float(row.get("trend_score", 0)) * 0.2,
        4)
    except:
        return 0.0
