from __future__ import annotations

from typing import Any

import pandas as pd

METRICS = {
    "revenue",
    "profit",
    "units",
    "orders",
    "avg_delivery_days",
    "cancellation_rate",
    "profit_margin",
}
DIMENSIONS = {"region", "category", "product", "status"}


class AnalysisError(ValueError):
    pass


def prepare_data(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "order_date",
        "order_id",
        "region",
        "category",
        "product",
        "units",
        "unit_price",
        "discount_pct",
        "cost_per_unit",
        "delivery_days",
        "status",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise AnalysisError(f"Missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise AnalysisError("The dataset has no rows.")

    result = frame.copy()
    result["order_date"] = pd.to_datetime(result["order_date"], errors="coerce")
    numeric = ["units", "unit_price", "discount_pct", "cost_per_unit", "delivery_days"]
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    result = result.dropna(subset=["order_date", "order_id", "units", "unit_price", "cost_per_unit"])
    result = result[result["units"] >= 0]
    result = result[result["unit_price"] >= 0]
    result = result[result["cost_per_unit"] >= 0]
    result["discount_pct"] = result["discount_pct"].fillna(0).clip(0, 100)
    result["revenue"] = result["units"] * result["unit_price"] * (1 - result["discount_pct"] / 100)
    result["profit"] = result["revenue"] - result["units"] * result["cost_per_unit"]
    result["is_cancelled"] = result["status"].astype(str).str.lower().eq("cancelled")
    if result.empty:
        raise AnalysisError("No valid rows remain after validating the required fields.")
    return result


def _metric_series(frame: pd.DataFrame, metric: str) -> pd.Series:
    if metric not in METRICS:
        raise AnalysisError(f"Unsupported metric '{metric}'. Choose from: {', '.join(sorted(METRICS))}.")
    if metric == "revenue":
        return frame["revenue"]
    if metric == "profit":
        return frame["profit"]
    if metric == "units":
        return frame["units"]
    if metric == "orders":
        return frame["order_id"]
    if metric == "avg_delivery_days":
        return frame["delivery_days"]
    if metric == "cancellation_rate":
        return frame["is_cancelled"].astype(float) * 100
    return frame["profit"] / frame["revenue"].replace(0, float("nan")) * 100


def _aggregate(frame: pd.DataFrame, metric: str) -> float | int | None:
    values = _metric_series(frame, metric)
    if metric == "orders":
        return int(values.nunique())
    if metric == "profit_margin":
        revenue = frame["revenue"].sum()
        value = frame["profit"].sum() / revenue * 100 if revenue else float("nan")
    elif metric in {"avg_delivery_days", "cancellation_rate"}:
        value = values.mean()
    else:
        value = values.sum()
    if pd.isna(value):
        return None
    return round(float(value), 2)


def profile_data(frame: pd.DataFrame) -> dict[str, Any]:
    status_counts = frame["status"].fillna("Unknown").astype(str).value_counts().to_dict()
    return {
        "valid_rows": int(len(frame)),
        "unique_orders": int(frame["order_id"].nunique()),
        "date_range": {
            "start": frame["order_date"].min().date().isoformat(),
            "end": frame["order_date"].max().date().isoformat(),
        },
        "regions": int(frame["region"].nunique()),
        "categories": int(frame["category"].nunique()),
        "missing_values": {key: int(value) for key, value in frame.isna().sum().items() if value},
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "revenue": _aggregate(frame, "revenue"),
        "profit": _aggregate(frame, "profit"),
        "average_delivery_days": _aggregate(frame, "avg_delivery_days"),
    }


def compare_metric(frame: pd.DataFrame, metric: str, group_by: str, limit: int = 10) -> dict[str, Any]:
    if group_by not in DIMENSIONS:
        raise AnalysisError(f"Unsupported grouping '{group_by}'. Choose from: {', '.join(sorted(DIMENSIONS))}.")
    if not 1 <= limit <= 20:
        raise AnalysisError("limit must be between 1 and 20.")
    _metric_series(frame, metric)
    rows = []
    for name, group in frame.groupby(group_by, dropna=False):
        rows.append({"group": str(name), "value": _aggregate(group, metric)})
    rows.sort(key=lambda item: float("-inf") if item["value"] is None else item["value"], reverse=True)
    return {"metric": metric, "group_by": group_by, "results": rows[:limit]}


def metric_trend(frame: pd.DataFrame, metric: str, granularity: str = "month") -> dict[str, Any]:
    if granularity not in {"month", "week"}:
        raise AnalysisError("granularity must be 'month' or 'week'.")
    _metric_series(frame, metric)
    data = frame.copy()
    data["period"] = data["order_date"].dt.to_period("M" if granularity == "month" else "W").astype(str)
    results = []
    for period, group in data.groupby("period", sort=True):
        results.append({"period": period, "value": _aggregate(group, metric)})
    for index, item in enumerate(results):
        previous = results[index - 1]["value"] if index else None
        current = item["value"]
        item["change_pct"] = None if previous in (None, 0) or current is None else round((current - previous) / abs(previous) * 100, 2)
    return {"metric": metric, "granularity": granularity, "results": results}


def delivery_risk(frame: pd.DataFrame, limit: int = 10) -> dict[str, Any]:
    if not 1 <= limit <= 20:
        raise AnalysisError("limit must be between 1 and 20.")
    data = frame.copy()
    rows = []
    for region, group in data.groupby("region", dropna=False):
        valid_days = group["delivery_days"].dropna()
        rows.append({
            "region": str(region),
            "orders": int(group["order_id"].nunique()),
            "average_delivery_days": round(float(valid_days.mean()), 2) if not valid_days.empty else None,
            "cancellation_rate_pct": round(float(group["is_cancelled"].mean() * 100), 2),
        })
    rows.sort(key=lambda item: (item["cancellation_rate_pct"], item["average_delivery_days"] or 0), reverse=True)
    return {"definition": "Regions are ranked by cancellation rate, then average delivery time. This is a screening signal, not proof of cause.", "regions": rows[:limit]}
