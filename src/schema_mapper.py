"""
DataPulse — E-Commerce Schema Mapper

Purpose
-------
Maps user-provided e-commerce dataset columns to a stable internal
DataPulse schema.

Important
---------
This module does NOT:
- perform EDA
- modify the user's dataframe
- clean business data
- train ML models
- calculate analytics

It only identifies which source columns correspond to the business
fields required by the DataPulse intelligence pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


# ============================================================
# DATAPULSE INTERNAL SCHEMA
# ============================================================

# Every downstream DataPulse module should use these canonical
# field names rather than depending directly on user's column names.

CANONICAL_FIELDS: Tuple[str, ...] = (
    # --------------------------------------------------------
    # Transaction / Sales
    # --------------------------------------------------------
    "order_id",
    "order_date",
    "order_status",
    "sales_amount",
    "quantity",
    "unit_price",
    "cost",
    "profit",

    # --------------------------------------------------------
    # Customer
    # --------------------------------------------------------
    "customer_id",
    "customer_name",
    "customer_email",
    "customer_age",
    "customer_gender",
    "customer_segment",
    "customer_city",
    "customer_state",
    "customer_country",
    "customer_since",

    # --------------------------------------------------------
    # Product
    # --------------------------------------------------------
    "product_id",
    "product_name",
    "category",
    "subcategory",
    "brand",

    # --------------------------------------------------------
    # Campaign Impact
    # --------------------------------------------------------
    "campaign_id",
    "campaign_name",
    "campaign_type",
    "campaign_start_date",
    "campaign_end_date",
    "campaign_cost",
    "discount",

    # --------------------------------------------------------
    # Market / Seller
    # --------------------------------------------------------
    "seller_id",
    "seller_name",
    "market",
    "region",

    # --------------------------------------------------------
    # Delivery / Operations
    # --------------------------------------------------------
    "shipping_date",
    "delivery_date",
    "promised_delivery_date",
    "delivery_status",
    "shipping_method",
    "return_flag",
    "return_date",

    # --------------------------------------------------------
    # Reviews / NLP
    # --------------------------------------------------------
    "review_id",
    "review_text",
    "review_rating",
    "review_date",

    # --------------------------------------------------------
    # Additional behavioral / ML fields
    # --------------------------------------------------------
    "payment_method",
    "device_type",
    "channel",
    "coupon_code",
)


# ============================================================
# FIELD GROUPS
# ============================================================

FIELD_GROUPS: Dict[str, Tuple[str, ...]] = {
    "sales": (
        "order_id",
        "order_date",
        "order_status",
        "sales_amount",
        "quantity",
        "unit_price",
        "cost",
        "profit",
    ),
    "customer": (
        "customer_id",
        "customer_name",
        "customer_email",
        "customer_age",
        "customer_gender",
        "customer_segment",
        "customer_city",
        "customer_state",
        "customer_country",
        "customer_since",
    ),
    "product": (
        "product_id",
        "product_name",
        "category",
        "subcategory",
        "brand",
    ),
    "campaign": (
        "campaign_id",
        "campaign_name",
        "campaign_type",
        "campaign_start_date",
        "campaign_end_date",
        "campaign_cost",
        "discount",
    ),
    "market_seller": (
        "seller_id",
        "seller_name",
        "market",
        "region",
        "customer_city",
        "customer_state",
        "customer_country",
    ),
    "delivery": (
        "shipping_date",
        "delivery_date",
        "promised_delivery_date",
        "delivery_status",
        "shipping_method",
        "return_flag",
        "return_date",
    ),
    "reviews": (
        "review_id",
        "review_text",
        "review_rating",
        "review_date",
    ),
    "behavioral_ml": (
        "payment_method",
        "device_type",
        "channel",
        "coupon_code",
    ),
}


# ============================================================
# REQUIRED FIELDS
# ============================================================

# These are the minimum business fields DataPulse generally needs
# for meaningful e-commerce intelligence.

CORE_REQUIRED_FIELDS: Tuple[str, ...] = (
    "order_id",
    "order_date",
    "sales_amount",
)


# ============================================================
# ALIASES
# ============================================================

# Each canonical field has common real-world dataset column names.

FIELD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "order_id": (
        "order_id",
        "orderid",
        "order_no",
        "order_number",
        "order",
        "transaction_id",
        "transactionid",
        "invoice_id",
        "invoice_no",
        "purchase_id",
    ),

    "order_date": (
        "order_date",
        "orderdate",
        "order_time",
        "order_datetime",
        "transaction_date",
        "transaction_datetime",
        "purchase_date",
        "date",
        "created_at",
        "created_date",
    ),

    "order_status": (
        "order_status",
        "status",
        "orderstate",
        "order_state",
        "fulfillment_status",
    ),

    "sales_amount": (
        "sales_amount",
        "sales",
        "sale",
        "revenue",
        "total_sales",
        "total_revenue",
        "revenue_amount",
        "order_value",
        "order_amount",
        "total_amount",
        "amount",
        "gmv",
        "gross_merchandise_value",
    ),

    "quantity": (
        "quantity",
        "qty",
        "units",
        "units_sold",
        "item_quantity",
        "order_quantity",
    ),

    "unit_price": (
        "unit_price",
        "price",
        "selling_price",
        "sale_price",
        "item_price",
        "product_price",
        "sellingprice",
    ),

    "cost": (
        "cost",
        "cost_price",
        "product_cost",
        "unit_cost",
        "purchase_cost",
        "cogs",
        "cost_of_goods_sold",
    ),

    "profit": (
        "profit",
        "profit_amount",
        "gross_profit",
        "net_profit",
        "profit_value",
    ),

    # --------------------------------------------------------
    # Customer
    # --------------------------------------------------------

    "customer_id": (
        "customer_id",
        "customerid",
        "customer_no",
        "customer_number",
        "client_id",
        "user_id",
        "buyer_id",
        "member_id",
    ),

    "customer_name": (
        "customer_name",
        "customer",
        "customername",
        "client_name",
        "buyer_name",
        "user_name",
        "name",
    ),

    "customer_email": (
        "customer_email",
        "email",
        "email_id",
        "customer_mail",
        "mail",
    ),

    "customer_age": (
        "customer_age",
        "age",
        "buyer_age",
        "user_age",
    ),

    "customer_gender": (
        "customer_gender",
        "gender",
        "sex",
        "buyer_gender",
        "user_gender",
    ),

    "customer_segment": (
        "customer_segment",
        "segment",
        "customer_group",
        "customer_type",
        "user_segment",
    ),

    "customer_city": (
        "customer_city",
        "city",
        "buyer_city",
        "user_city",
        "shipping_city",
        "billing_city",
    ),

    "customer_state": (
        "customer_state",
        "state",
        "province",
        "buyer_state",
        "user_state",
        "shipping_state",
        "billing_state",
    ),

    "customer_country": (
        "customer_country",
        "country",
        "buyer_country",
        "user_country",
        "shipping_country",
        "billing_country",
    ),

    "customer_since": (
        "customer_since",
        "customer_start_date",
        "registration_date",
        "signup_date",
        "join_date",
        "customer_created_at",
    ),

    # --------------------------------------------------------
    # Product
    # --------------------------------------------------------

    "product_id": (
        "product_id",
        "productid",
        "product_no",
        "product_number",
        "item_id",
        "sku",
        "sku_id",
    ),

    "product_name": (
        "product_name",
        "product",
        "productname",
        "item_name",
        "item",
        "sku_name",
    ),

    "category": (
        "category",
        "product_category",
        "item_category",
        "category_name",
    ),

    "subcategory": (
        "subcategory",
        "sub_category",
        "product_subcategory",
        "sub_category_name",
    ),

    "brand": (
        "brand",
        "brand_name",
        "manufacturer",
        "product_brand",
    ),

    # --------------------------------------------------------
    # Campaign
    # --------------------------------------------------------

    "campaign_id": (
        "campaign_id",
        "campaignid",
        "campaign_no",
        "promotion_id",
        "promo_id",
    ),

    "campaign_name": (
        "campaign_name",
        "campaign",
        "campaignname",
        "promotion_name",
        "promo_name",
        "marketing_campaign",
    ),

    "campaign_type": (
        "campaign_type",
        "promotion_type",
        "promo_type",
        "marketing_type",
    ),

    "campaign_start_date": (
        "campaign_start_date",
        "campaign_start",
        "promotion_start_date",
        "promo_start_date",
    ),

    "campaign_end_date": (
        "campaign_end_date",
        "campaign_end",
        "promotion_end_date",
        "promo_end_date",
    ),

    "campaign_cost": (
        "campaign_cost",
        "marketing_cost",
        "campaign_spend",
        "ad_spend",
        "advertising_cost",
        "promotion_cost",
    ),

    "discount": (
        "discount",
        "discount_amount",
        "discount_value",
        "discount_percent",
        "discount_percentage",
        "discount_rate",
        "markdown",
        "promotion_discount",
    ),

    # --------------------------------------------------------
    # Seller / Market
    # --------------------------------------------------------

    "seller_id": (
        "seller_id",
        "sellerid",
        "seller_no",
        "vendor_id",
        "merchant_id",
        "store_id",
    ),

    "seller_name": (
        "seller_name",
        "seller",
        "sellername",
        "vendor_name",
        "merchant_name",
        "store_name",
    ),

    "market": (
        "market",
        "market_name",
        "marketplace",
        "marketplace_name",
    ),

    "region": (
        "region",
        "region_name",
        "sales_region",
        "geographic_region",
        "territory",
    ),

    # --------------------------------------------------------
    # Delivery
    # --------------------------------------------------------

    "shipping_date": (
        "shipping_date",
        "ship_date",
        "shipped_date",
        "dispatch_date",
        "shipment_date",
    ),

    "delivery_date": (
        "delivery_date",
        "delivered_date",
        "actual_delivery_date",
        "received_date",
    ),

    "promised_delivery_date": (
        "promised_delivery_date",
        "expected_delivery_date",
        "estimated_delivery_date",
        "delivery_due_date",
        "promised_date",
    ),

    "delivery_status": (
        "delivery_status",
        "shipping_status",
        "shipment_status",
        "delivery_state",
        "fulfillment_status",
    ),

    "shipping_method": (
        "shipping_method",
        "shipping_type",
        "delivery_method",
        "delivery_type",
        "ship_method",
        "carrier_service",
    ),

    "return_flag": (
        "return_flag",
        "returned",
        "is_returned",
        "return",
        "returned_flag",
        "return_status",
    ),

    "return_date": (
        "return_date",
        "returned_date",
        "refund_date",
        "return_created_at",
    ),

    # --------------------------------------------------------
    # Reviews
    # --------------------------------------------------------

    "review_id": (
        "review_id",
        "reviewid",
        "review_no",
        "feedback_id",
    ),

    "review_text": (
        "review_text",
        "review",
        "review_comment",
        "comment",
        "comments",
        "feedback",
        "feedback_text",
        "customer_review",
        "customer_feedback",
    ),

    "review_rating": (
        "review_rating",
        "rating",
        "review_score",
        "star_rating",
        "stars",
        "customer_rating",
    ),

    "review_date": (
        "review_date",
        "reviewed_date",
        "feedback_date",
        "rating_date",
    ),

    # --------------------------------------------------------
    # Behavioral
    # --------------------------------------------------------

    "payment_method": (
        "payment_method",
        "payment_type",
        "payment",
        "pay_method",
        "mode_of_payment",
    ),

    "device_type": (
        "device_type",
        "device",
        "platform_device",
        "customer_device",
    ),

    "channel": (
        "channel",
        "sales_channel",
        "order_channel",
        "marketing_channel",
        "source_channel",
    ),

    "coupon_code": (
        "coupon_code",
        "coupon",
        "promo_code",
        "promotion_code",
        "voucher_code",
    ),
}


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass(frozen=True)
class ColumnMapping:
    """Represents one canonical DataPulse field mapping."""

    canonical_field: str
    source_column: Optional[str]
    confidence: float
    match_type: str
    available: bool


@dataclass
class SchemaMappingResult:
    """Stable schema contract consumed by downstream modules."""

    mappings: Dict[str, ColumnMapping]
    mapped_columns: Dict[str, str]
    unmapped_source_columns: List[str]
    missing_core_fields: List[str]
    available_fields: List[str]
    field_groups: Dict[str, Dict[str, bool]]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mappings": {
                key: asdict(value)
                for key, value in self.mappings.items()
            },
            "mapped_columns": dict(self.mapped_columns),
            "unmapped_source_columns": list(
                self.unmapped_source_columns
            ),
            "missing_core_fields": list(
                self.missing_core_fields
            ),
            "available_fields": list(
                self.available_fields
            ),
            "field_groups": {
                group: dict(fields)
                for group, fields in self.field_groups.items()
            },
            "warnings": list(self.warnings),
        }


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_column_name(name: Any) -> str:
    """
    Convert a source column name into a comparison-friendly form.

    Examples
    --------
    "Order ID"       -> "order id"
    "order_id"       -> "order id"
    "Order-ID"       -> "order id"
    "Customer Email" -> "customer email"
    """

    text = str(name).strip().lower()

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def compact_name(name: Any) -> str:
    """Remove separators for stronger alias comparison."""

    return re.sub(
        r"[^a-z0-9]",
        "",
        normalize_column_name(name),
    )


# ============================================================
# MATCHING
# ============================================================

def _exact_alias_match(
    source_column: str,
    aliases: Iterable[str],
) -> bool:

    normalized_source = normalize_column_name(
        source_column
    )

    compact_source = compact_name(
        source_column
    )

    for alias in aliases:
        if normalized_source == normalize_column_name(alias):
            return True

        if compact_source == compact_name(alias):
            return True

    return False


def _fuzzy_similarity(
    source_column: str,
    alias: str,
) -> float:

    source_normalized = compact_name(
        source_column
    )

    alias_normalized = compact_name(
        alias
    )

    if not source_normalized or not alias_normalized:
        return 0.0

    return SequenceMatcher(
        None,
        source_normalized,
        alias_normalized,
    ).ratio()


def _best_match(
    source_columns: List[str],
    canonical_field: str,
) -> Tuple[
    Optional[str],
    float,
    str,
]:

    aliases = FIELD_ALIASES.get(
        canonical_field,
        (canonical_field,),
    )

    # --------------------------------------------------------
    # 1. Exact alias match
    # --------------------------------------------------------

    exact_candidates = []

    for source_column in source_columns:
        if _exact_alias_match(
            source_column,
            aliases,
        ):
            exact_candidates.append(source_column)

    if exact_candidates:
        # Prefer the shortest exact match because names such as
        # "order_id" are usually safer than verbose variants.
        best = sorted(
            exact_candidates,
            key=lambda x: (
                len(normalize_column_name(x)),
                len(x),
            ),
        )[0]

        return best, 1.0, "exact"

    # --------------------------------------------------------
    # 2. Fuzzy matching
    # --------------------------------------------------------

    candidates: List[
        Tuple[float, str]
    ] = []

    for source_column in source_columns:
        best_alias_score = max(
            (
                _fuzzy_similarity(
                    source_column,
                    alias,
                )
                for alias in aliases
            ),
            default=0.0,
        )

        candidates.append(
            (
                best_alias_score,
                source_column,
            )
        )

    if not candidates:
        return None, 0.0, "none"

    candidates.sort(
        key=lambda item: (
            item[0],
            -len(item[1]),
        ),
        reverse=True,
    )

    score, column = candidates[0]

    # Conservative threshold.
    # We don't want an incorrect mapping to silently damage
    # downstream analytics or ML.
    if score >= 0.78:
        return column, score, "fuzzy"

    return None, score, "none"


# ============================================================
# FIELD TYPE HEURISTICS
# ============================================================

def _is_numeric_series(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def _is_datetime_like(series: pd.Series) -> bool:

    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    if series.empty:
        return False

    sample = series.dropna().head(100)

    if sample.empty:
        return False

    converted = pd.to_datetime(
        sample,
        errors="coerce",
    )

    valid_ratio = converted.notna().mean()

    return bool(valid_ratio >= 0.75)


def _is_text_series(series: pd.Series) -> bool:

    return (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    )


# ============================================================
# TYPE COMPATIBILITY
# ============================================================

def _type_compatibility(
    df: pd.DataFrame,
    canonical_field: str,
    source_column: Optional[str],
) -> float:

    if source_column is None:
        return 0.0

    if source_column not in df.columns:
        return 0.0

    series = df[source_column]

    numeric_fields = {
        "sales_amount",
        "quantity",
        "unit_price",
        "cost",
        "profit",
        "customer_age",
        "campaign_cost",
        "discount",
        "review_rating",
    }

    date_fields = {
        "order_date",
        "customer_since",
        "campaign_start_date",
        "campaign_end_date",
        "shipping_date",
        "delivery_date",
        "promised_delivery_date",
        "return_date",
        "review_date",
    }

    text_fields = {
        "customer_name",
        "customer_email",
        "customer_gender",
        "customer_segment",
        "customer_city",
        "customer_state",
        "customer_country",
        "product_name",
        "category",
        "subcategory",
        "brand",
        "campaign_name",
        "campaign_type",
        "seller_name",
        "market",
        "region",
        "delivery_status",
        "shipping_method",
        "review_text",
        "payment_method",
        "device_type",
        "channel",
        "coupon_code",
    }

    id_fields = {
        "order_id",
        "customer_id",
        "product_id",
        "campaign_id",
        "seller_id",
        "review_id",
    }

    if canonical_field in numeric_fields:
        return 1.0 if _is_numeric_series(series) else 0.65

    if canonical_field in date_fields:
        return 1.0 if _is_datetime_like(series) else 0.55

    if canonical_field in text_fields:
        return 1.0 if _is_text_series(series) else 0.60

    if canonical_field in id_fields:
        # IDs are frequently stored as either string or integer.
        return 1.0

    return 1.0


# ============================================================
# MAIN SCHEMA MAPPER
# ============================================================

def map_schema(
    df: pd.DataFrame,
    fuzzy_threshold: float = 0.78,
) -> Dict[str, Any]:
    """
    Map an e-commerce dataframe into the DataPulse schema.

    Parameters
    ----------
    df:
        Analysis-ready dataframe.

    fuzzy_threshold:
        Minimum fuzzy similarity required for automatic mapping.

    Returns
    -------
    dict
        Stable mapping contract for validator and downstream modules.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "map_schema() expects a pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "Cannot map schema for an empty dataset."
        )

    if len(df.columns) == 0:
        raise ValueError(
            "Cannot map schema for a dataset without columns."
        )

    source_columns = [
        str(column)
        for column in df.columns
    ]

    mappings: Dict[
        str,
        ColumnMapping,
    ] = {}

    mapped_columns: Dict[str, str] = {}

    used_source_columns = set()

    warnings: List[str] = []

    # --------------------------------------------------------
    # Map every canonical field
    # --------------------------------------------------------

    for canonical_field in CANONICAL_FIELDS:

        source_column, score, match_type = _best_match(
            source_columns,
            canonical_field,
        )

        # Don't allow one source column to automatically satisfy
        # unrelated canonical fields.
        if (
            source_column is not None
            and source_column in used_source_columns
        ):
            alternative_candidates = [
                column
                for column in source_columns
                if column not in used_source_columns
            ]

            alternative, alternative_score, alternative_type = (
                _best_match(
                    alternative_candidates,
                    canonical_field,
                )
            )

            if alternative is not None:
                source_column = alternative
                score = alternative_score
                match_type = alternative_type
            else:
                source_column = None
                score = 0.0
                match_type = "none"

        # Respect caller's threshold.
        if (
            match_type == "fuzzy"
            and score < fuzzy_threshold
        ):
            source_column = None
            score = 0.0
            match_type = "none"

        if source_column is not None:
            compatibility = _type_compatibility(
                df,
                canonical_field,
                source_column,
            )

            # Slightly adjust confidence using datatype compatibility.
            final_confidence = round(
                (score * 0.75) + (compatibility * 0.25),
                4,
            )

            used_source_columns.add(
                source_column
            )

            mapped_columns[
                canonical_field
            ] = source_column

            mappings[
                canonical_field
            ] = ColumnMapping(
                canonical_field=canonical_field,
                source_column=source_column,
                confidence=final_confidence,
                match_type=match_type,
                available=True,
            )

        else:

            mappings[
                canonical_field
            ] = ColumnMapping(
                canonical_field=canonical_field,
                source_column=None,
                confidence=0.0,
                match_type="none",
                available=False,
            )

    # --------------------------------------------------------
    # Available fields
    # --------------------------------------------------------

    available_fields = [
        field
        for field, mapping in mappings.items()
        if mapping.available
    ]

    # --------------------------------------------------------
    # Missing core fields
    # --------------------------------------------------------

    missing_core_fields = [
        field
        for field in CORE_REQUIRED_FIELDS
        if not mappings[field].available
    ]

    # --------------------------------------------------------
    # Field groups
    # --------------------------------------------------------

    field_groups: Dict[
        str,
        Dict[str, bool],
    ] = {}

    for group_name, fields in FIELD_GROUPS.items():

        field_groups[group_name] = {
            field: mappings[field].available
            for field in fields
        }

    # --------------------------------------------------------
    # Unmapped source columns
    # --------------------------------------------------------

    unmapped_source_columns = [
        column
        for column in source_columns
        if column not in used_source_columns
    ]

    # --------------------------------------------------------
    # Warnings
    # --------------------------------------------------------

    for field in available_fields:

        mapping = mappings[field]

        if (
            mapping.match_type == "fuzzy"
            and mapping.confidence < 0.90
        ):
            warnings.append(
                f"'{mapping.source_column}' was mapped to "
                f"'{field}' using fuzzy matching "
                f"(confidence={mapping.confidence:.2f})."
            )

    if missing_core_fields:

        warnings.append(
            "Missing core DataPulse fields: "
            + ", ".join(missing_core_fields)
            + "."
        )

    if unmapped_source_columns:

        warnings.append(
            "Some source columns were not mapped to a "
            "DataPulse field: "
            + ", ".join(unmapped_source_columns[:10])
            + (
                "..."
                if len(unmapped_source_columns) > 10
                else "."
            )
        )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = SchemaMappingResult(
        mappings=mappings,
        mapped_columns=mapped_columns,
        unmapped_source_columns=unmapped_source_columns,
        missing_core_fields=missing_core_fields,
        available_fields=available_fields,
        field_groups=field_groups,
        warnings=warnings,
    )

    return result.to_dict()


# ============================================================
# CONVENIENCE HELPERS
# ============================================================

def get_source_column(
    mapping: Dict[str, Any],
    canonical_field: str,
) -> Optional[str]:
    """Return the original source column for a canonical field."""

    mapped_columns = mapping.get(
        "mapped_columns",
        {},
    )

    return mapped_columns.get(
        canonical_field
    )


def has_field(
    mapping: Dict[str, Any],
    canonical_field: str,
) -> bool:
    """Check whether a canonical field exists in the dataset."""

    return canonical_field in mapping.get(
        "mapped_columns",
        {},
    )


def get_available_fields(
    mapping: Dict[str, Any],
) -> List[str]:
    """Return all successfully mapped canonical fields."""

    return list(
        mapping.get(
            "available_fields",
            [],
        )
    )


def get_missing_core_fields(
    mapping: Dict[str, Any],
) -> List[str]:
    """Return missing core fields."""

    return list(
        mapping.get(
            "missing_core_fields",
            [],
        )
    )


def get_group_status(
    mapping: Dict[str, Any],
    group_name: str,
) -> Dict[str, bool]:
    """Return availability of fields in a DataPulse module group."""

    return dict(
        mapping.get(
            "field_groups",
            {},
        ).get(
            group_name,
            {},
        )
    )


def is_group_available(
    mapping: Dict[str, Any],
    group_name: str,
    minimum_fields: int = 1,
) -> bool:
    """
    Determine whether enough fields exist to activate a module.

    This does not decide whether the dataset is mathematically
    sufficient for a particular analysis. That responsibility
    belongs to the validator/analytics layer.
    """

    group = get_group_status(
        mapping,
        group_name,
    )

    available_count = sum(
        bool(value)
        for value in group.values()
    )

    return available_count >= minimum_fields


# ============================================================
# DISPLAY / DEBUG HELPERS
# ============================================================

def mapping_summary(
    mapping: Dict[str, Any],
) -> pd.DataFrame:
    """
    Convert schema mapping into a dataframe useful for debugging
    and future UI display.
    """

    rows = []

    raw_mappings = mapping.get(
        "mappings",
        {},
    )

    for canonical_field in CANONICAL_FIELDS:

        item = raw_mappings.get(
            canonical_field,
            {},
        )

        rows.append(
            {
                "DataPulse Field": canonical_field,
                "Source Column": item.get(
                    "source_column"
                ),
                "Confidence": item.get(
                    "confidence",
                    0.0,
                ),
                "Match Type": item.get(
                    "match_type",
                    "none",
                ),
                "Available": item.get(
                    "available",
                    False,
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    "CANONICAL_FIELDS",
    "FIELD_GROUPS",
    "CORE_REQUIRED_FIELDS",
    "FIELD_ALIASES",
    "ColumnMapping",
    "SchemaMappingResult",
    "normalize_column_name",
    "map_schema",
    "get_source_column",
    "has_field",
    "get_available_fields",
    "get_missing_core_fields",
    "get_group_status",
    "is_group_available",
    "mapping_summary",
]