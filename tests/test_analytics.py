import pandas as pd
import pytest

from opslens.analytics import AnalysisError, compare_metric, metric_trend, prepare_data, profile_data


@pytest.fixture
def sample_frame():
    return pd.DataFrame([
        {"order_date": "2026-01-03", "order_id": "A1", "region": "North", "category": "Home", "product": "Lamp", "units": 2, "unit_price": 100, "discount_pct": 0, "cost_per_unit": 40, "delivery_days": 2, "status": "Delivered"},
        {"order_date": "2026-01-15", "order_id": "A2", "region": "South", "category": "Tech", "product": "Mouse", "units": 1, "unit_price": 200, "discount_pct": 10, "cost_per_unit": 80, "delivery_days": 4, "status": "Cancelled"},
        {"order_date": "2026-02-02", "order_id": "A3", "region": "North", "category": "Home", "product": "Lamp", "units": 3, "unit_price": 100, "discount_pct": 0, "cost_per_unit": 40, "delivery_days": 3, "status": "Delivered"},
    ])


def test_prepare_data_calculates_revenue_and_profit(sample_frame):
    frame = prepare_data(sample_frame)
    assert frame["revenue"].tolist() == [200, 180, 300]
    assert frame["profit"].tolist() == [120, 100, 180]


def test_profile_counts_unique_orders(sample_frame):
    result = profile_data(prepare_data(sample_frame))
    assert result["valid_rows"] == 3
    assert result["unique_orders"] == 3
    assert result["revenue"] == 680


def test_compare_metric_ranks_groups(sample_frame):
    result = compare_metric(prepare_data(sample_frame), "revenue", "region")
    assert result["results"][0] == {"group": "North", "value": 500}


def test_trend_has_period_change(sample_frame):
    result = metric_trend(prepare_data(sample_frame), "revenue")
    assert len(result["results"]) == 2
    assert result["results"][1]["change_pct"] == -21.05


def test_missing_required_column_is_rejected(sample_frame):
    with pytest.raises(AnalysisError, match="Missing required columns"):
        prepare_data(sample_frame.drop(columns=["region"]))


def test_unsupported_metric_is_rejected(sample_frame):
    with pytest.raises(AnalysisError, match="Unsupported metric"):
        compare_metric(prepare_data(sample_frame), "customer_happiness", "region")


def test_invalid_grouping_is_rejected(sample_frame):
    with pytest.raises(AnalysisError, match="Unsupported grouping"):
        compare_metric(prepare_data(sample_frame), "revenue", "email")


def test_empty_dataset_is_rejected(sample_frame):
    with pytest.raises(AnalysisError, match="no rows"):
        prepare_data(sample_frame.iloc[0:0])
