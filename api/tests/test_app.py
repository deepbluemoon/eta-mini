import pytest
from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)


def get_eta(payload):
    """/eta 는 GET 이지만 JSON body 를 받으므로 request 로 보낸다."""
    return client.request("GET", "/eta", json=payload)


def test_health():
    """/health 가 200 과 status, model 을 돌려주는지 확인한다."""
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok", "model": "rule"}


def test_eta_response_shape():
    """/eta 응답이 minutes(float) 와 model(str) 만 담는지 확인한다."""
    res = get_eta({"distance_km": 1, "pending_orders": 1})
    assert res.status_code == 200
    body = res.json()
    assert set(body) == {"minutes", "model"}
    assert isinstance(body["minutes"], float)
    assert body["model"] == "rule"


@pytest.mark.parametrize(
    ("distance_km", "pending_orders", "expected"),
    [
        (0, 0, 10.0),
        (2.5, 3, 23.5),
        (1000, 500, 4010.0),
    ],
)
def test_eta_known_values(distance_km, pending_orders, expected):
    """정상·경계(0)·큰 값 입력에 대해 예상 시간을 돌려주는지 확인한다."""
    res = get_eta({"distance_km": distance_km, "pending_orders": pending_orders})
    assert res.status_code == 200
    assert res.json()["minutes"] == pytest.approx(expected)


@pytest.mark.parametrize(
    "payload",
    [
        {"distance_km": -0.1, "pending_orders": 0},
        {"distance_km": 0, "pending_orders": -1},
        {"distance_km": "far", "pending_orders": 0},
        {"distance_km": 0, "pending_orders": "many"},
        {"distance_km": 0, "pending_orders": 1.5},
        {"distance_km": 1},
        {"pending_orders": 1},
        {},
    ],
)
def test_eta_invalid_input_returns_422(payload):
    """음수, 숫자가 아닌 값, 정수가 아닌 주문 수, 누락된 필드는 422 를 돌려주는지 확인한다."""
    assert get_eta(payload).status_code == 422
