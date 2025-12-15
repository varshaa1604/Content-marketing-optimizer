# sentiment_analyzer.py (FULLY FIXED & CLEAN VERSION)

import logging
from typing import List, Union, Dict
from textblob import TextBlob


# ------------------------------------------------------
# Optional imports (HuggingFace, langdetect)
# ------------------------------------------------------
try:
    from transformers import pipeline
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

try:
    from langdetect import detect
    LANG_AVAILABLE = True
except Exception:
    LANG_AVAILABLE = False


# ------------------------------------------------------
# Safe optional imports for integrations
# ------------------------------------------------------
try:
    from app.integrations.social_ingestor import SocialIngestor
except Exception:
    SocialIngestor = None

try:
    from app.integrations.sheets_connector import append_row
except Exception:
    def append_row(*args, **kwargs):
        return False


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_senti_model = None
_emotion_model = None


# ------------------------------------------------------
# HF Model loaders
# ------------------------------------------------------
def _init_sentiment_model():
    return pipeline("sentiment-analysis")


def _init_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None
    )


# ------------------------------------------------------
# Language detection
# ------------------------------------------------------
def detect_language(text: str) -> str:
    if not LANG_AVAILABLE:
        return "unknown"
    try:
        return detect(text)
    except Exception:
        return "unknown"


# ------------------------------------------------------
# Fallback sentiment (TextBlob)
# ------------------------------------------------------
def fallback_sentiment(text: str) -> Dict:
    polarity = TextBlob(text).sentiment.polarity

    if polarity >= 0.05:
        label = "POSITIVE"
    elif polarity <= -0.05:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"

    return {
        "label": label,
        "score": abs(polarity),
        "polarity": polarity
    }


# ------------------------------------------------------
# Simplify emotion model output
# ------------------------------------------------------
def simplify_emotion_output(raw_output: List[Dict]) -> Dict:
    try:
        return {x["label"]: float(x["score"]) for x in raw_output}
    except Exception:
        return {}


# ------------------------------------------------------
# Comment-level analysis
# ------------------------------------------------------
def analyze_post_comments(post_id: str) -> Dict:

    # No SocialIngestor available — return safe defaults
    if SocialIngestor is None:
        return {
            "post_id": post_id,
            "avg_sentiment": 0.5,
            "avg_polarity": 0.0,
            "avg_toxicity": 0.0,
            "labels": {},
            "samples": []
        }

    ingestor = SocialIngestor()
    if not hasattr(ingestor, "fetch_post_comments"):
        return {
            "post_id": post_id,
            "avg_sentiment": 0.5,
            "avg_polarity": 0.0,
            "avg_toxicity": 0.0,
            "labels": {},
            "samples": []
        }

    comments = ingestor.fetch_post_comments(post_id)
    if not comments:
        return {
            "post_id": post_id,
            "avg_sentiment": 0.5,
            "avg_polarity": 0.0,
            "avg_toxicity": 0.0,
            "labels": {},
            "samples": []
        }

    results = analyze_sentiment(comments)

    labels = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    for r in results:
        labels[r["sentiment_label"]] += 1

    avg_sent = sum(r["sentiment_score"] for r in results) / len(results)
    avg_pol = sum(r["polarity"] for r in results) / len(results)

    toxic_vals = [r["emotions"].get("anger", 0) for r in results]
    avg_toxic = sum(toxic_vals) / len(toxic_vals)

    try:
        append_row("comment_sentiment", [
            post_id, avg_sent, avg_pol, avg_toxic, labels
        ])
    except Exception:
        pass

    return {
        "post_id": post_id,
        "avg_sentiment": round(avg_sent, 4),
        "avg_polarity": round(avg_pol, 4),
        "avg_toxicity": round(avg_toxic, 4),
        "labels": labels,
        "samples": results
    }


# ------------------------------------------------------
# MAIN SENTIMENT ANALYSIS FUNCTION
# ------------------------------------------------------
def analyze_sentiment(texts: Union[str, List[str]]) -> List[Dict]:

    if isinstance(texts, str):
        texts = [texts]

    global _senti_model, _emotion_model, HF_AVAILABLE

    # Try loading HF models only once
    if HF_AVAILABLE:
        try:
            if _senti_model is None:
                _senti_model = _init_sentiment_model()
            if _emotion_model is None:
                _emotion_model = _init_emotion_model()
        except Exception:
            logger.error("HF model initialization failed — switching to fallback mode.")
            HF_AVAILABLE = False

    results = []

    for text in texts:
        lang = detect_language(text)

        # SENTIMENT --------------------
        if HF_AVAILABLE:
            try:
                pred = _senti_model(text)[0]
                label = pred["label"].upper()
                score = float(pred["score"])
                polarity = TextBlob(text).sentiment.polarity
            except Exception:
                s = fallback_sentiment(text)
                label, score, polarity = s["label"], s["score"], s["polarity"]
        else:
            s = fallback_sentiment(text)
            label, score, polarity = s["label"], s["score"], s["polarity"]

        # Normalize score
        norm_score = (1 - score) if label.startswith("NEG") else score

        # EMOTIONS --------------------
        emotions = {}
        if HF_AVAILABLE:
            try:
                emo_raw = _emotion_model(text)[0]
                emotions = simplify_emotion_output(emo_raw)
            except Exception:
                emotions = {}

        entry = {
            "text": text,
            "sentiment_label": label,
            "sentiment_score": round(norm_score, 4),
            "polarity": polarity,
            "emotions": emotions,
            "language": lang,
            "trend_score": 0.0
        }

        # Log to sheet (non-blocking)
        try:
            append_row("sentiment_results", [
                text[:80] + "...", label, norm_score, polarity
            ])
        except Exception:
            pass

        results.append(entry)

    return results
