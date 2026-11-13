import math

import pytest

from api.predictors.rule import estimate_delivery_minutes


@pytest.mark.parametrize(
    ("distance_km", "pending_orders", "expected"),
    [
        (0, 0, 10.0),
        (1, 0, 13.0),
        (0, 1, 12.0),
        (2.5, 3, 23.5),
        (1000, 500, 4010.0),
    ],
)
def test_known_inputs(distance_km, pending_orders, expected):
    """알려진 입력에 대해 10 + 3*거리 + 2*주문수 를 돌려주는지 확인한다."""
    assert estimate_delivery_minutes(distance_km, pending_orders) == pytest.approx(
        expected
    )


def test_returns_float():
    """정수 입력이어도 float 을 돌려주는지 확인한다."""
    assert isinstance(estimate_delivery_minutes(1, 1), float)


def test_float_distance_precision():
    """소수 거리도 1km 당 3분으로 계산하는지 확인한다."""
    assert estimate_delivery_minutes(0.1, 0) == pytest.approx(10.3)


def test_negative_inputs_not_validated_current_behavior():
    """현재 동작: 함수 자체는 음수를 검증하지 않고 공식대로 계산한다(검증은 API 계층 몫)."""
    assert estimate_delivery_minutes(-1, -1) == pytest.approx(5.0)


def test_nan_propagates_current_behavior():
    """현재 동작: NaN 입력은 NaN 을 돌려준다."""
    assert math.isnan(estimate_delivery_minutes(float("nan"), 0))


def test_overflow_returns_inf_current_behavior():
    """현재 동작: 아주 큰 거리는 inf 로 overflow 된다(API 의 minutes=null 결함의 원인)."""
    assert math.isinf(estimate_delivery_minutes(1e308, 0))
