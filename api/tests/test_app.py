import importlib

import pytest
from fastapi.testclient import TestClient

import api.app as app_module
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


@pytest.mark.parametrize(
    ("distance_km", "pending_orders", "expected"),
    [
        (1, 0, 13.0),
        (0, 1, 12.0),
        (0.0001, 0, 10.0003),
        (-0.0, 0, 10.0),
    ],
)
def test_eta_coefficients_and_near_zero(distance_km, pending_orders, expected):
    """거리·주문 계수를 하나씩, 그리고 0 바로 위 값과 -0.0 을 정상 처리하는지 확인한다."""
    res = get_eta({"distance_km": distance_km, "pending_orders": pending_orders})
    assert res.status_code == 200
    assert res.json()["minutes"] == pytest.approx(expected)


@pytest.mark.parametrize(
    ("payload", "bad_field"),
    [
        ({"distance_km": None, "pending_orders": 0}, "distance_km"),
        ({"distance_km": 0, "pending_orders": None}, "pending_orders"),
        ({"distance_km": [1], "pending_orders": 0}, "distance_km"),
        ({"distance_km": 0, "pending_orders": {"n": 1}}, "pending_orders"),
        ({"distance_km": -0.1, "pending_orders": 0}, "distance_km"),
        ({"distance_km": 0, "pending_orders": -1}, "pending_orders"),
    ],
)
def test_eta_422_points_to_bad_field(payload, bad_field):
    """null·배열·객체·음수 값은 422 이고 detail 의 loc 이 해당 필드를 가리키는지 확인한다."""
    res = get_eta(payload)
    assert res.status_code == 422
    locs = [err["loc"] for err in res.json()["detail"]]
    assert ["body", bad_field] in locs


@pytest.mark.parametrize("content", ["not json", "[]", '"text"', "1"])
def test_eta_malformed_body_returns_422(content):
    """깨진 JSON 이나 객체가 아닌 JSON body 는 422 를 돌려주는지 확인한다."""
    res = client.request(
        "GET", "/eta", content=content, headers={"content-type": "application/json"}
    )
    assert res.status_code == 422


def test_eta_without_body_returns_422():
    """body 없이 /eta 를 호출하면 422 를 돌려주는지 확인한다."""
    assert client.request("GET", "/eta").status_code == 422


def test_eta_query_params_are_not_accepted():
    """값을 query string 으로만 보내면 body 가 없으므로 422 를 돌려주는지 확인한다."""
    res = client.get("/eta", params={"distance_km": 1, "pending_orders": 1})
    assert res.status_code == 422


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/eta"),
        ("PUT", "/eta"),
        ("DELETE", "/eta"),
        ("POST", "/health"),
    ],
)
def test_wrong_method_returns_405(method, path):
    """정의되지 않은 HTTP 메서드는 405 를 돌려주는지 확인한다."""
    res = client.request(method, path, json={"distance_km": 1, "pending_orders": 1})
    assert res.status_code == 405


def test_unknown_path_returns_404():
    """없는 경로는 404 를 돌려주는지 확인한다."""
    assert client.get("/does-not-exist").status_code == 404


# --- 현재 동작 고정: 정책이 정해지면 기대값을 바꾼다 ---


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"distance_km": "1.5", "pending_orders": 0}, 14.5),
        ({"distance_km": 0, "pending_orders": "2"}, 14.0),
        ({"distance_km": 0, "pending_orders": 2.0}, 14.0),
        ({"distance_km": True, "pending_orders": 0}, 13.0),
        ({"distance_km": 0, "pending_orders": True}, 12.0),
    ],
)
def test_eta_lax_coercion_current_behavior(payload, expected):
    """현재 동작: 숫자 문자열, 정수 값의 float, bool 을 숫자로 변환해 200 을 돌려준다."""
    res = get_eta(payload)
    assert res.status_code == 200
    assert res.json()["minutes"] == pytest.approx(expected)


def test_eta_extra_fields_ignored_current_behavior():
    """현재 동작: 정의되지 않은 필드는 무시하고 200 을 돌려준다."""
    res = get_eta({"distance_km": 1, "pending_orders": 1, "weather": "rain"})
    assert res.status_code == 200
    assert res.json() == {"minutes": 15.0, "model": "rule"}


def test_eta_huge_int_orders_current_behavior():
    """현재 동작: 주문 수에 상한이 없어 아주 큰 정수도 그대로 계산한다."""
    res = get_eta({"distance_km": 0, "pending_orders": 10**30})
    assert res.status_code == 200
    assert res.json()["minutes"] == pytest.approx(2e30)


# --- 알려진 결함: 앱이 고쳐지면 XPASS 로 실패하므로 마커를 지운다 ---

lenient_client = TestClient(app, raise_server_exceptions=False)


def send_raw_eta(content):
    """JSON 표준 밖의 토큰(NaN, Infinity)을 보내려고 body 를 문자열 그대로 보낸다."""
    return lenient_client.request(
        "GET", "/eta", content=content, headers={"content-type": "application/json"}
    )


@pytest.mark.xfail(
    strict=True,
    reason="결함: NaN 은 검증에서 걸리지만 422 본문을 JSON 으로 만들다 ValueError 로 500 이 난다",
)
def test_eta_nan_distance_returns_422():
    """distance_km 가 NaN 이면 500 이 아니라 422 를 돌려주는지 확인한다."""
    res = send_raw_eta('{"distance_km": NaN, "pending_orders": 0}')
    assert res.status_code == 422


@pytest.mark.xfail(
    strict=True,
    reason="결함: Infinity 가 ge=0 검증을 통과해 200 과 minutes=null 을 돌려준다",
)
def test_eta_infinite_distance_returns_422():
    """distance_km 가 Infinity 이면 422 를 돌려주는지 확인한다."""
    res = send_raw_eta('{"distance_km": Infinity, "pending_orders": 0}')
    assert res.status_code == 422


@pytest.mark.xfail(
    strict=True,
    reason="결함: 3*1e308 이 inf 로 overflow 되어 200 과 minutes=null 을 돌려준다",
)
def test_eta_overflowing_distance_never_returns_null_minutes():
    """계산이 overflow 되는 큰 거리는 422 이거나, 200 이면 minutes 가 숫자인지 확인한다."""
    res = lenient_client.request("GET", "/eta", json={"distance_km": 1e308, "pending_orders": 0})
    if res.status_code != 422:
        assert res.status_code == 200
        assert isinstance(res.json()["minutes"], float)


def test_unknown_model_impl_fails_at_import(monkeypatch):
    """MODEL_IMPL 이 없는 구현체 이름이면 앱 import 시점에 ModuleNotFoundError 가 나는지 확인한다."""
    monkeypatch.setenv("MODEL_IMPL", "does_not_exist")
    try:
        with pytest.raises(ModuleNotFoundError):
            importlib.reload(app_module)
    finally:
        monkeypatch.undo()
        importlib.reload(app_module)
