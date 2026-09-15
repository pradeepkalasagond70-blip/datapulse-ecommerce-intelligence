"""
DataPulse - Review & NLP Analytics

Purpose
-------
Analyzes customer reviews and converts unstructured feedback into
business intelligence.

Covers:
- Review volume
- Rating distribution
- Average rating
- Positive / negative / neutral sentiment
- Keyword extraction
- Product review performance
- Category review performance
- Rating vs revenue
- Sentiment vs rating
- Negative feedback themes
- Review quality signals
- Business recommendations

Stable interface
----------------
run_review_nlp(df, mapping) -> dict

Notes
-----
This module intentionally uses lightweight, dependency-safe NLP.
It does not require an external model download or internet connection.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

POSITIVE_WORDS = {
    "amazing",
    "awesome",
    "best",
    "better",
    "brilliant",
    "clean",
    "comfortable",
    "easy",
    "excellent",
    "fast",
    "favorite",
    "friendly",
    "good",
    "great",
    "happy",
    "helpful",
    "impressive",
    "love",
    "loved",
    "nice",
    "perfect",
    "quick",
    "recommend",
    "recommended",
    "satisfied",
    "smooth",
    "super",
    "useful",
    "value",
    "wonderful",
    "worth",
}

NEGATIVE_WORDS = {
    "bad",
    "broken",
    "cancel",
    "canceled",
    "delay",
    "delayed",
    "difficult",
    "disappointed",
    "disappointing",
    "dirty",
    "expensive",
    "hate",
    "hated",
    "late",
    "missing",
    "poor",
    "problem",
    "refund",
    "return",
    "rude",
    "slow",
    "terrible",
    "unhappy",
    "unhelpful",
    "unreliable",
    "waste",
    "worst",
    "wrong",
}

STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}

NEGATION_WORDS = {
    "not",
    "never",
    "no",
    "none",
    "neither",
    "hardly",
    "without",
}


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def _get_source_column(
    mapping: Dict[str, Any],
    canonical_field: str,
) -> Optional[str]:
    """Return the source column mapped to a canonical field."""

    if not mapping:
        return None

    mappings = mapping.get("mappings", {})
    value = mappings.get(canonical_field)

    if isinstance(value, dict):
        return value.get("source_column") or value.get("column")

    if isinstance(value, str):
        return value

    return None


def _safe_numeric(
    series: pd.Series,
) -> pd.Series:
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def _safe_text(
    series: pd.Series,
    default: str = "",
) -> pd.Series:
    result = series.astype("string").str.strip()

    result = result.replace(
        {
            "nan": pd.NA,
            "None": pd.NA,
            "null": pd.NA,
        }
    )

    return result.fillna(default)


def _empty(
    columns: list[str],
) -> pd.DataFrame:
    return pd.DataFrame(
        columns=columns
    )


# ---------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------

def _tokenize(
    text: str,
) -> List[str]:
    """Convert text into normalized word tokens."""

    if not text:
        return []

    text = text.lower()

    tokens = re.findall(
        r"[a-zA-Z][a-zA-Z']+",
        text,
    )

    cleaned = []

    for token in tokens:
        token = token.lower().strip("'")

        if (
            len(token) < 3
            or token in STOPWORDS
        ):
            continue

        cleaned.append(token)

    return cleaned


def _clean_text(
    text: str,
) -> str:
    """Normalize review text."""

    if not text:
        return ""

    text = str(text)

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ---------------------------------------------------------------------
# Sentiment
# ---------------------------------------------------------------------

def _sentiment_score(
    text: str,
) -> float:
    """
    Lightweight lexicon sentiment score.

    Score is approximately normalized to -1 to +1.
    """

    tokens = _tokenize(text)

    if not tokens:
        return 0.0

    score = 0.0
    sentiment_terms = 0

    for index, token in enumerate(tokens):
        value = 0

        if token in POSITIVE_WORDS:
            value = 1

        elif token in NEGATIVE_WORDS:
            value = -1

        if value == 0:
            continue

        # Basic negation handling.
        previous_tokens = tokens[
            max(0, index - 2):index
        ]

        if any(
            word in NEGATION_WORDS
            for word in previous_tokens
        ):
            value *= -1

        score += value
        sentiment_terms += 1

    if sentiment_terms == 0:
        return 0.0

    normalized = score / max(
        sentiment_terms,
        1,
    )

    return float(
        np.clip(
            normalized,
            -1,
            1,
        )
    )


def _sentiment_label(
    score: float,
) -> str:
    """Convert sentiment score into a business-friendly label."""

    if score >= 0.20:
        return "Positive"

    if score <= -0.20:
        return "Negative"

    return "Neutral"


# ---------------------------------------------------------------------
# Prepare dataframe
# ---------------------------------------------------------------------

def _prepare_dataframe(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
) -> pd.DataFrame:
    """Create standardized internal review dataframe."""

    fields = [
        "review_id",
        "review_text",
        "review_rating",
        "review_date",
        "product_id",
        "product_name",
        "category",
        "customer_id",
        "seller_id",
        "seller_name",
        "region",
        "market",
        "order_id",
    ]

    working = pd.DataFrame(
        index=df.index
    )

    for field in fields:
        source_column = _get_source_column(
            mapping,
            field,
        )

        if (
            source_column
            and source_column in df.columns
        ):
            working[field] = df[
                source_column
            ]

    if "review_text" in working.columns:
        working["review_text"] = _safe_text(
            working["review_text"]
        ).apply(_clean_text)

    if "review_rating" in working.columns:
        working["review_rating"] = _safe_numeric(
            working["review_rating"]
        )

    if "review_date" in working.columns:
        working["review_date"] = pd.to_datetime(
            working["review_date"],
            errors="coerce",
        )

    for field in [
        "review_id",
        "product_id",
        "product_name",
        "category",
        "customer_id",
        "seller_id",
        "seller_name",
        "region",
        "market",
        "order_id",
    ]:
        if field in working.columns:
            working[field] = _safe_text(
                working[field]
            )

    return working


# ---------------------------------------------------------------------
# Review-level sentiment
# ---------------------------------------------------------------------

def _analyze_reviews(
    working: pd.DataFrame,
) -> pd.DataFrame:
    """Add sentiment score and label to every review."""

    result = working.copy()

    if "review_text" not in result.columns:
        return result

    result["sentiment_score"] = (
        result["review_text"]
        .apply(_sentiment_score)
    )

    result["sentiment"] = (
        result["sentiment_score"]
        .apply(_sentiment_label)
    )

    result["review_word_count"] = (
        result["review_text"]
        .apply(
            lambda text: len(
                _tokenize(text)
            )
        )
    )

    return result


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def _build_summary(
    working: pd.DataFrame,
) -> Dict[str, Any]:
    """Build review and sentiment KPIs."""

    review_count = len(working)

    if "review_rating" in working.columns:
        ratings = working[
            "review_rating"
        ].dropna()

        average_rating = (
            float(ratings.mean())
            if not ratings.empty
            else 0.0
        )

        median_rating = (
            float(ratings.median())
            if not ratings.empty
            else 0.0
        )

        rating_5_rate = (
            float(
                (ratings == 5).mean()
                * 100
            )
            if not ratings.empty
            else 0.0
        )

        low_rating_rate = (
            float(
                (ratings <= 2).mean()
                * 100
            )
            if not ratings.empty
            else 0.0
        )

    else:
        average_rating = 0.0
        median_rating = 0.0
        rating_5_rate = 0.0
        low_rating_rate = 0.0

    if "sentiment" in working.columns:
        positive_count = int(
            (
                working["sentiment"]
                == "Positive"
            ).sum()
        )

        neutral_count = int(
            (
                working["sentiment"]
                == "Neutral"
            ).sum()
        )

        negative_count = int(
            (
                working["sentiment"]
                == "Negative"
            ).sum()
        )

    else:
        positive_count = 0
        neutral_count = 0
        negative_count = 0

    sentiment_total = (
        positive_count
        + neutral_count
        + negative_count
    )

    return {
        "review_count": int(review_count),
        "average_rating": round(
            average_rating,
            2,
        ),
        "median_rating": round(
            median_rating,
            2,
        ),
        "five_star_rate": round(
            rating_5_rate,
            2,
        ),
        "low_rating_rate": round(
            low_rating_rate,
            2,
        ),
        "positive_reviews": positive_count,
        "neutral_reviews": neutral_count,
        "negative_reviews": negative_count,
        "positive_rate": round(
            (
                positive_count
                / sentiment_total
                * 100
            )
            if sentiment_total
            else 0.0,
            2,
        ),
        "neutral_rate": round(
            (
                neutral_count
                / sentiment_total
                * 100
            )
            if sentiment_total
            else 0.0,
            2,
        ),
        "negative_rate": round(
            (
                negative_count
                / sentiment_total
                * 100
            )
            if sentiment_total
            else 0.0,
            2,
        ),
    }


# ---------------------------------------------------------------------
# Rating distribution
# ---------------------------------------------------------------------

def _build_rating_distribution(
    working: pd.DataFrame,
) -> pd.DataFrame:
    """Build rating distribution."""

    if "review_rating" not in working.columns:
        return _empty(
            [
                "rating",
                "reviews",
                "share_pct",
            ]
        )

    ratings = (
        working["review_rating"]
        .dropna()
        .round()
    )

    if ratings.empty:
        return _empty(
            [
                "rating",
                "reviews",
                "share_pct",
            ]
        )

    result = (
        ratings.value_counts()
        .sort_index()
        .rename_axis("rating")
        .reset_index(
            name="reviews"
        )
    )

    total = result["reviews"].sum()

    result["share_pct"] = (
        result["reviews"]
        / total
        * 100
        if total
        else 0
    )

    return result


# ---------------------------------------------------------------------
# Sentiment distribution
# ---------------------------------------------------------------------

def _build_sentiment_distribution(
    working: pd.DataFrame,
) -> pd.DataFrame:
    """Build sentiment distribution."""

    if "sentiment" not in working.columns:
        return _empty(
            [
                "sentiment",
                "reviews",
                "share_pct",
                "average_rating",
            ]
        )

    grouped = (
        working.groupby(
            "sentiment",
            dropna=False,
        )
        .agg(
            reviews=("sentiment", "size"),
            average_rating=(
                "review_rating",
                "mean",
            )
            if "review_rating"
            in working.columns
            else (
                "sentiment",
                lambda x: 0.0,
            ),
            average_sentiment_score=(
                "sentiment_score",
                "mean",
            ),
        )
        .reset_index()
    )

    total = grouped["reviews"].sum()

    grouped["share_pct"] = (
        grouped["reviews"]
        / total
        * 100
        if total
        else 0
    )

    return grouped.sort_values(
        "reviews",
        ascending=False,
    ).reset_index(
        drop=True
    )


# ---------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------

def _extract_keywords(
    texts: pd.Series,
    top_n: int = 30,
) -> pd.DataFrame:
    """Extract frequent meaningful words."""

    counter = Counter()

    for text in texts:
        tokens = _tokenize(
            str(text)
        )

        counter.update(tokens)

    if not counter:
        return _empty(
            [
                "keyword",
                "frequency",
            ]
        )

    result = pd.DataFrame(
        counter.most_common(top_n),
        columns=[
            "keyword",
            "frequency",
        ],
    )

    return result


# ---------------------------------------------------------------------
# Positive keywords
# ---------------------------------------------------------------------

def _extract_sentiment_keywords(
    working: pd.DataFrame,
    sentiment: str,
    top_n: int = 20,
) -> pd.DataFrame:
    """Extract common words from positive or negative reviews."""

    if (
        "review_text" not in working.columns
        or "sentiment" not in working.columns
    ):
        return _empty(
            [
                "keyword",
                "frequency",
            ]
        )

    texts = working.loc[
        working["sentiment"] == sentiment,
        "review_text",
    ]

    return _extract_keywords(
        texts,
        top_n=top_n,
    )


# ---------------------------------------------------------------------
# Product reviews
# ---------------------------------------------------------------------

def _build_product_reviews(
    working: pd.DataFrame,
) -> pd.DataFrame:
    """Analyze reviews by product."""

    product_field = None

    if "product_name" in working.columns:
        product_field = "product_name"

    elif "product_id" in working.columns:
        product_field = "product_id"

    if product_field is None:
        return _empty(
            [
                "product",
                "reviews",
                "average_rating",
                "positive_rate",
                "negative_rate",
                "average_sentiment_score",
            ]
        )

    aggregations = {
        "reviews": (
            "review_text",
            "size",
        ),
        "average_sentiment_score": (
            "sentiment_score",
            "mean",
        ),
    }

    if "review_rating" in working.columns:
        aggregations["average_rating"] = (
            "review_rating",
            "mean",
        )

    result = (
        working.groupby(
            product_field,
            dropna=False,
        )
        .agg(**aggregations)
        .reset_index()
        .rename(
            columns={
                product_field: "product"
            }
        )
    )

    if "sentiment" in working.columns:
        positive = (
            working["sentiment"]
            == "Positive"
        )

        negative = (
            working["sentiment"]
            == "Negative"
        )

        positive_rate = (
            positive.groupby(
                working[product_field]
            )
            .mean()
            * 100
        )

        negative_rate = (
            negative.groupby(
                working[product_field]
            )
            .mean()
            * 100
        )

        result["positive_rate"] = (
            result["product"]
            .map(positive_rate)
            .fillna(0)
        )

        result["negative_rate"] = (
            result["product"]
            .map(negative_rate)
            .fillna(0)
        )

    else:
        result["positive_rate"] = 0.0
        result["negative_rate"] = 0.0

    return result.sort_values(
        "reviews",
        ascending=False,
    ).reset_index(
        drop=True
    )


# ---------------------------------------------------------------------
# Category reviews
# ---------------------------------------------------------------------

def _build_category_reviews(
    working: pd.DataFrame,
) -> pd.DataFrame:
    """Analyze reviews by category."""

    if "category" not in working.columns:
        return _empty(
            [
                "category",
                "reviews",
                "average_rating",
                "positive_rate",
                "negative_rate",
                "average_sentiment_score",
            ]
        )

    aggregations = {
        "reviews": (
            "review_text",
            "size",
        ),
        "average_sentiment_score": (
            "sentiment_score",
            "mean",
        ),
    }

    if "review_rating" in working.columns:
        aggregations["average_rating"] = (
            "review_rating",
            "mean",
        )

    result = (
        working.groupby(
            "category",
            dropna=False,
        )
        .agg(**aggregations)
        .reset_index()
    )

    if "sentiment" in working.columns:
        positive_rate = (
            (
                working["sentiment"]
                == "Positive"
            )
            .groupby(
                working["category"]
            )
            .mean()
            * 100
        )

        negative_rate = (
            (
                working["sentiment"]
                == "Negative"
            )
            .groupby(
                working["category"]
            )
            .mean()
            * 100
        )

        result["positive_rate"] = (
            result["category"]
            .map(positive_rate)
            .fillna(0)
        )

        result["negative_rate"] = (
            result["category"]
            .map(negative_rate)
            .fillna(0)
        )

    else:
        result["positive_rate"] = 0.0
        result["negative_rate"] = 0.0

    return result.sort_values(
        "reviews",
        ascending=False,
    ).reset_index(
        drop=True
    )


# ---------------------------------------------------------------------
# Rating vs sentiment
# ---------------------------------------------------------------------

def _build_rating_sentiment_matrix(
    working: pd.DataFrame,
) -> pd.DataFrame:
    """Compare sentiment classifications against customer ratings."""

    if (
        "review_rating" not in working.columns
        or "sentiment" not in working.columns
    ):
        return _empty(
            [
                "rating",
                "sentiment",
                "reviews",
                "share_pct",
            ]
        )

    result = (
        working.dropna(
            subset=["review_rating"]
        )
        .groupby(
            [
                "review_rating",
                "sentiment",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="reviews"
        )
        .rename(
            columns={
                "review_rating": "rating",
            }
        )
    )

    total = result["reviews"].sum()

    result["share_pct"] = (
        result["reviews"]
        / total
        * 100
        if total
        else 0
    )

    return result.sort_values(
        [
            "rating",
            "reviews",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(
        drop=True
    )


# ---------------------------------------------------------------------
# Negative feedback themes
# ---------------------------------------------------------------------

def _build_negative_themes(
    working: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identify common words in negative reviews.

    This is a keyword-based theme detector rather than a generative
    topic-modeling system.
    """

    negative = working[
        working["sentiment"] == "Negative"
    ] if "sentiment" in working.columns else pd.DataFrame()

    if negative.empty:
        return _empty(
            [
                "theme",
                "frequency",
            ]
        )

    keywords = _extract_keywords(
        negative["review_text"],
        top_n=30,
    )

    if keywords.empty:
        return keywords

    theme_map = {
        "delivery": {
            "delay",
            "delayed",
            "late",
            "shipping",
            "delivery",
            "arrive",
            "arrived",
        },
        "quality": {
            "broken",
            "poor",
            "quality",
            "damaged",
            "wrong",
            "defective",
        },
        "price": {
            "expensive",
            "price",
            "cost",
            "value",
        },
        "service": {
            "rude",
            "unhelpful",
            "support",
            "service",
            "staff",
        },
        "returns": {
            "return",
            "refund",
            "replacement",
            "replace",
        },
        "product_experience": {
            "difficult",
            "problem",
            "missing",
            "slow",
            "bad",
        },
    }

    theme_counts = Counter()

    for _, row in keywords.iterrows():
        word = row["keyword"]
        frequency = int(
            row["frequency"]
        )

        for theme, words in theme_map.items():
            if word in words:
                theme_counts[theme] += frequency

    if not theme_counts:
        return _empty(
            [
                "theme",
                "frequency",
            ]
        )

    return pd.DataFrame(
        theme_counts.most_common(),
        columns=[
            "theme",
            "frequency",
        ],
    )


# ---------------------------------------------------------------------
# Review quality
# ---------------------------------------------------------------------

def _build_review_quality(
    working: pd.DataFrame,
) -> Dict[str, Any]:
    """Measure basic review completeness and quality signals."""

    if "review_text" not in working.columns:
        return {
            "available": False,
            "text_coverage_pct": 0.0,
            "empty_reviews": 0,
            "short_reviews": 0,
            "average_word_count": 0.0,
        }

    total = len(working)

    empty_reviews = int(
        (
            working["review_text"]
            .str.strip()
            .eq("")
        ).sum()
    )

    short_reviews = int(
        (
            working["review_word_count"]
            <= 2
        ).sum()
    )

    populated = total - empty_reviews

    average_word_count = float(
        working["review_word_count"].mean()
    ) if total else 0.0

    return {
        "available": True,
        "text_coverage_pct": round(
            populated / total * 100
            if total
            else 0.0,
            2,
        ),
        "empty_reviews": empty_reviews,
        "short_reviews": short_reviews,
        "average_word_count": round(
            average_word_count,
            2,
        ),
    }


# ---------------------------------------------------------------------
# Review trends
# ---------------------------------------------------------------------

def _build_review_trend(
    working: pd.DataFrame,
) -> pd.DataFrame:
    """Build monthly review and sentiment trend."""

    if (
        "review_date" not in working.columns
        or "review_text" not in working.columns
    ):
        return _empty(
            [
                "month",
                "reviews",
                "average_rating",
                "average_sentiment_score",
                "positive_rate",
                "negative_rate",
            ]
        )

    data = working.dropna(
        subset=["review_date"]
    ).copy()

    if data.empty:
        return _empty(
            [
                "month",
                "reviews",
                "average_rating",
                "average_sentiment_score",
                "positive_rate",
                "negative_rate",
            ]
        )

    data["month"] = (
        data["review_date"]
        .dt.to_period("M")
        .astype(str)
    )

    result = (
        data.groupby("month")
        .agg(
            reviews=(
                "review_text",
                "size",
            ),
            average_rating=(
                "review_rating",
                "mean",
            )
            if "review_rating"
            in data.columns
            else (
                "review_text",
                lambda x: 0.0,
            ),
            average_sentiment_score=(
                "sentiment_score",
                "mean",
            ),
        )
        .reset_index()
    )

    if "sentiment" in data.columns:
        positive_rate = (
            (
                data["sentiment"]
                == "Positive"
            )
            .groupby(
                data["month"]
            )
            .mean()
            * 100
        )

        negative_rate = (
            (
                data["sentiment"]
                == "Negative"
            )
            .groupby(
                data["month"]
            )
            .mean()
            * 100
        )

        result["positive_rate"] = (
            result["month"]
            .map(positive_rate)
            .fillna(0)
        )

        result["negative_rate"] = (
            result["month"]
            .map(negative_rate)
            .fillna(0)
        )

    else:
        result["positive_rate"] = 0.0
        result["negative_rate"] = 0.0

    return result.sort_values(
        "month"
    ).reset_index(
        drop=True
    )


# ---------------------------------------------------------------------
# Review recommendations
# ---------------------------------------------------------------------

def _build_recommendations(
    summary: Dict[str, Any],
    negative_themes: pd.DataFrame,
    product_reviews: pd.DataFrame,
) -> list[Dict[str, Any]]:
    """Generate business recommendations from review intelligence."""

    recommendations = []

    negative_rate = summary.get(
        "negative_rate",
        0.0,
    )

    if negative_rate >= 25:
        recommendations.append(
            {
                "priority": "High",
                "type": "Customer Experience",
                "message": (
                    f"Negative review rate is "
                    f"{negative_rate:.1f}%. "
                    "Investigate recurring customer pain points."
                ),
            }
        )

    elif negative_rate >= 15:
        recommendations.append(
            {
                "priority": "Medium",
                "type": "Customer Experience",
                "message": (
                    f"Negative review rate is "
                    f"{negative_rate:.1f}%. "
                    "Monitor recurring complaints."
                ),
            }
        )

    if not negative_themes.empty:
        top_theme = negative_themes.iloc[0]

        recommendations.append(
            {
                "priority": "High",
                "type": "Top Feedback Theme",
                "message": (
                    f"'{top_theme['theme']}' is the most prominent "
                    "negative feedback theme. Prioritize root-cause "
                    "analysis for this area."
                ),
            }
        )

    if not product_reviews.empty:
        low_rated = product_reviews[
            (
                product_reviews["reviews"]
                >= product_reviews["reviews"].median()
            )
            & (
                product_reviews["average_rating"]
                < product_reviews["average_rating"].median()
            )
        ]

        if not low_rated.empty:
            recommendations.append(
                {
                    "priority": "Medium",
                    "type": "Product Experience",
                    "message": (
                        f"{len(low_rated)} product(s) have meaningful "
                        "review volume but below-median ratings. "
                        "Review product quality and customer experience."
                    ),
                }
            )

    return recommendations


# ---------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------

def _build_capabilities(
    working: pd.DataFrame,
) -> Dict[str, bool]:
    """Expose available NLP/review analytics."""

    return {
        "review_analysis": (
            "review_text" in working.columns
        ),
        "sentiment_analysis": (
            "review_text" in working.columns
        ),
        "rating_analysis": (
            "review_rating" in working.columns
        ),
        "keyword_analysis": (
            "review_text" in working.columns
        ),
        "product_review_analysis": (
            "review_text" in working.columns
            and (
                "product_name" in working.columns
                or "product_id" in working.columns
            )
        ),
        "category_review_analysis": (
            "review_text" in working.columns
            and "category" in working.columns
        ),
        "review_trend_analysis": (
            "review_text" in working.columns
            and "review_date" in working.columns
        ),
    }


# ---------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------

def run_review_nlp(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run Review & NLP Analytics.

    Parameters
    ----------
    df:
        Analysis-ready dataframe.

    mapping:
        Schema mapping generated by schema_mapper.py.

    Returns
    -------
    dict
        Stable Review/NLP analytics contract.
    """

    if (
        df is None
        or not isinstance(df, pd.DataFrame)
        or df.empty
    ):
        return {
            "available": False,
            "summary": {},
            "rating_distribution": pd.DataFrame(),
            "sentiment_distribution": pd.DataFrame(),
            "keywords": pd.DataFrame(),
            "positive_keywords": pd.DataFrame(),
            "negative_keywords": pd.DataFrame(),
            "product_reviews": pd.DataFrame(),
            "category_reviews": pd.DataFrame(),
            "rating_sentiment_matrix": pd.DataFrame(),
            "negative_themes": pd.DataFrame(),
            "review_trend": pd.DataFrame(),
            "review_quality": {},
            "recommendations": [],
            "capabilities": {},
            "warnings": [
                "A non-empty pandas DataFrame is required."
            ],
        }

    working = _prepare_dataframe(
        df,
        mapping,
    )

    capabilities = _build_capabilities(
        working
    )

    if not capabilities["review_analysis"]:
        return {
            "available": False,
            "summary": {},
            "rating_distribution": pd.DataFrame(),
            "sentiment_distribution": pd.DataFrame(),
            "keywords": pd.DataFrame(),
            "positive_keywords": pd.DataFrame(),
            "negative_keywords": pd.DataFrame(),
            "product_reviews": pd.DataFrame(),
            "category_reviews": pd.DataFrame(),
            "rating_sentiment_matrix": pd.DataFrame(),
            "negative_themes": pd.DataFrame(),
            "review_trend": pd.DataFrame(),
            "review_quality": {},
            "recommendations": [],
            "capabilities": capabilities,
            "warnings": [
                "Review text is unavailable. "
                "Review & NLP Analytics cannot run."
            ],
        }

    # Review-level NLP
    working = _analyze_reviews(
        working
    )

    # Summary
    summary = _build_summary(
        working
    )

    # Distributions
    rating_distribution = (
        _build_rating_distribution(
            working
        )
    )

    sentiment_distribution = (
        _build_sentiment_distribution(
            working
        )
    )

    # Keywords
    keywords = _extract_keywords(
        working["review_text"],
        top_n=30,
    )

    positive_keywords = (
        _extract_sentiment_keywords(
            working,
            "Positive",
            top_n=20,
        )
    )

    negative_keywords = (
        _extract_sentiment_keywords(
            working,
            "Negative",
            top_n=20,
        )
    )

    # Dimensional analysis
    product_reviews = (
        _build_product_reviews(
            working
        )
    )

    category_reviews = (
        _build_category_reviews(
            working
        )
    )

    rating_sentiment_matrix = (
        _build_rating_sentiment_matrix(
            working
        )
    )

    negative_themes = (
        _build_negative_themes(
            working
        )
    )

    review_trend = (
        _build_review_trend(
            working
        )
    )

    review_quality = (
        _build_review_quality(
            working
        )
    )

    recommendations = (
        _build_recommendations(
            summary,
            negative_themes,
            product_reviews,
        )
    )

    warnings = []

    if not capabilities["rating_analysis"]:
        warnings.append(
            "Review rating is unavailable; "
            "rating-based analytics are limited."
        )

    if not capabilities["product_review_analysis"]:
        warnings.append(
            "Product fields are unavailable; "
            "product-level review analysis is limited."
        )

    if not capabilities["category_review_analysis"]:
        warnings.append(
            "Category field is unavailable; "
            "category-level review analysis is limited."
        )

    if not capabilities["review_trend_analysis"]:
        warnings.append(
            "Review date is unavailable; "
            "review trend analysis is limited."
        )

    return {
        "available": True,
        "summary": summary,
        "rating_distribution": rating_distribution,
        "sentiment_distribution": sentiment_distribution,
        "keywords": keywords,
        "positive_keywords": positive_keywords,
        "negative_keywords": negative_keywords,
        "product_reviews": product_reviews,
        "category_reviews": category_reviews,
        "rating_sentiment_matrix": rating_sentiment_matrix,
        "negative_themes": negative_themes,
        "review_trend": review_trend,
        "review_quality": review_quality,
        "recommendations": recommendations,
        "capabilities": capabilities,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------

run_nlp_analytics = run_review_nlp
run_review_analytics = run_review_nlp