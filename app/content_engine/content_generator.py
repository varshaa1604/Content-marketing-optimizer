"""
Simple, safe content_generator.py

Purpose:
- Produce short social media post variants given simple inputs.
- Avoid external dependencies so Streamlit UI works even offline.
- Return list of dicts: {"text": str, "meta": {...}}
"""

import random
import html
from typing import List, Dict, Optional

# small helper utilities
def _clean_hashtags(tags: List[str]) -> List[str]:
    out = []
    for t in tags:
        if not t:
            continue
        t = t.strip()
        if not t:
            continue
        if not t.startswith("#"):
            t = "#" + t
        # remove punctuation at end
        t = t.rstrip(".,!?:;")
        out.append(t)
    # de-duplicate preserving order
    seen = set()
    uniq = []
    for t in out:
        tl = t.lower()
        if tl not in seen:
            seen.add(tl)
            uniq.append(t)
    return uniq

def _move_hashtags_to_end(text: str, hashtags: List[str]) -> str:
    # remove any hashtags inside text then append curated hashtags at end
    words = text.split()
    words_no_tags = [w for w in words if not w.startswith("#")]
    suffix = " ".join(hashtags[:2])  # keep up to 2 hashtags
    if suffix:
        return (" ".join(words_no_tags)).strip() + "  " + suffix
    return " ".join(words_no_tags).strip()


# simple template-based generator tuned for Twitter-style short posts (~40-70 words)
def _make_variant(topic: str,
                  platform: str,
                  audience: str,
                  tone: str,
                  keywords: List[str],
                  word_count: int,
                  variant_idx: int,
                  trends: Optional[List[str]] = None) -> Dict:

    # basic hooks and CTAs
    hooks = [
        "Stop scrolling —",
        "Big idea:",
        "Quick truth:",
        "Heads up marketers:",
        "Ready for growth?"
    ]
    energizers = [
        "Use AI to scale creativity and save time.",
        "Automate repetitive tasks and double down on what matters.",
        "Turn data into decisions, not noise.",
        "Personalize at scale without the busywork.",
    ]
    cta = "Follow for more marketing insights!"

    hook = random.choice(hooks)
    energizer = random.choice(energizers)

    # blend keywords and trend keywords
    clean_tags = _clean_hashtags([k for k in keywords if k])
    trend_tags = _clean_hashtags([t for t in (trends or []) if t])
    # prefer including at most 1 trending tag + 1 keyword tag
    chosen_tags = []
    if trend_tags:
        chosen_tags.append(trend_tags[0])
    if clean_tags:
        # pick a hashtag that's not duplicate
        for t in clean_tags:
            if t not in chosen_tags:
                chosen_tags.append(t)
                break

    # Build human text (approximate words)
    # Grow/shrink sentences based on desired word_count, but keep short for Twitter
    base_sentences = [
        f"{hook} {topic} is rewriting how {audience} find and convert customers.",
        f"{energizer} {cta}",
        "Short, practical tips > long strategy documents."
    ]

    # Mix sentences, but ensure natural flow
    if tone and tone.lower() in ("funny", "playful", "witty"):
        base_sentences.insert(1, "Yes — AI can be clever *and* human.")
    elif tone and tone.lower() in ("professional", "serious"):
        base_sentences.insert(1, "Focus on measurable outcomes and test fast.")
    else:
        # positive / neutral default
        base_sentences.insert(1, "Small experiments now = big wins later.")

    # Choose number of sentences so text ~ word_count but not exceeding platform norms
    # Twitter best practice: keep under ~70 words; ignore if user asks > 100.
    max_words = max(20, min(word_count, 100))
    combined = " ".join(base_sentences)

    # trim to approximately max_words by cutting sentences if needed
    words = combined.split()
    if len(words) > max_words:
        # keep first N words and append ellipsis
        combined = " ".join(words[:max_words-1]) + "…"

    # ensure no duplicated hashtags inside the sentence; move to end
    final_text = _move_hashtags_to_end(combined, chosen_tags)

    # small humanization: unescape any html entities
    final_text = html.unescape(final_text)

    meta = {
        "topic": topic,
        "platform": platform,
        "audience": audience,
        "injected_keywords": chosen_tags,
        "trend_score": 0,
        "trend_insights": {}
    }

    return {"text": final_text.strip(), "meta": meta}


# Public function expected by Streamlit app
def generate_final_variations(
    topic: str,
    platform: str,
    keywords: List[str],
    audience: str,
    tone: str = "positive",
    n: int = 2,
    word_count: int = 50,
    past_metrics: Optional[Dict] = None
) -> List[Dict]:
    """
    Safe, deterministic (but varied) content generator.
    Returns list of dicts: {"text": "...", "meta": {...}}
    """

    # guard against bad inputs
    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(",") if k.strip()]

    # small deterministic randomness seed (so outputs differ per run)
    random.seed(42 + n + (len(topic) if topic else 0))

    # optional: simple local trend fetcher stub (overrideable)
    trends = []
    try:
        # if you have TrendFetcher available, you can replace this section
        from app.integrations.trend_fetcher import TrendFetcher
        tf = TrendFetcher()
        trends = tf.fetch_google_global_trends() or []
    except Exception:
        trends = []

    variants = []
    for i in range(n):
        v = _make_variant(topic=topic,
                          platform=platform,
                          audience=audience,
                          tone=tone,
                          keywords=keywords,
                          word_count=word_count,
                          variant_idx=i+1,
                          trends=trends)
        variants.append(v)

    return variants


# Quick local test when module executed directly
if __name__ == "__main__":
    outs = generate_final_variations(
        topic="AI in Marketing",
        platform="Twitter",
        keywords=["#AI", "#Marketing"],
        audience="digital marketers",
        tone="positive",
        n=2,
        word_count=50
    )
    for i, o in enumerate(outs, 1):
        print(f"\nVariant {i}\n{o['text']}\nMETA: {o['meta']}\n")
