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
    assert estimate_delivery_minutes(distance_km, pending_orders) == pytest.approx(expected)


def test_returns_float():
    """정수 입력이어도 float 을 돌려주는지 확인한다."""
    assert isinstance(estimate_delivery_minutes(1, 1), float)
