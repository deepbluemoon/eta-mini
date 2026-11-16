"""다른 언어 서비스(Go, Rust)의 예측이 Python API 와 같은지 비교한다.

pytest 테스트가 아니다. 서버가 모두 떠 있어야 하므로 직접 실행한다.

    PY_URL=http://127.0.0.1:8080 \\
    TARGETS=go=http://127.0.0.1:8081,rust=http://127.0.0.1:8082 \\
    python -m training.check_parity

기준(PY_URL)은 학습한 원본 모델(MODEL_IMPL=sklearn)로 띄운 Python API 다.
대상마다 최대 오차가 TOLERANCE 분 미만이면 PASS. 하나라도 FAIL 이면 exit 1.
"""

import json
import os
import random
import sys
import urllib.error
import urllib.request

TOLERANCE = 1e-4  # 분
PY_URL = os.getenv("PY_URL", "http://127.0.0.1:8080")
TARGETS = os.getenv("TARGETS", "go=http://127.0.0.1:8081,rust=http://127.0.0.1:8082")
BASE_MODEL = "sklearn"


def parse_targets(spec: str) -> dict[str, str]:
    """'go=http://...,rust=http://...' 를 {이름: URL} 로 바꾼다."""
    targets = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        name, sep, url = item.partition("=")
        if not sep or not name or not url:
            raise ValueError(
                f"TARGETS 형식이 잘못됐다: {item!r} (예: go=http://127.0.0.1:8081)"
            )
        targets[name.strip()] = url.strip().rstrip("/")
    return targets


def make_inputs() -> list[tuple[float, int]]:
    """고정 입력 3건 + seed 0 무작위 20건 (거리 0.5~8km, 주문 0~10건)."""
    rng = random.Random(0)
    fixed = [(3.0, 2), (0.0, 0), (8.0, 10)]
    rand = [(rng.uniform(0.5, 8.0), rng.randint(0, 10)) for _ in range(20)]
    return fixed + rand


def get_json(url: str, body: dict | None = None) -> dict:
    """GET 요청을 보내고 JSON 응답을 돌려준다. /eta 는 GET 에 JSON body 를 받는다."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method="GET", headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.load(resp)


def predict(base_url: str, distance_km: float, pending_orders: int) -> float:
    """서비스의 /eta 를 불러 예상 시간(분)을 돌려준다."""
    out = get_json(
        f"{base_url}/eta",
        {"distance_km": distance_km, "pending_orders": pending_orders},
    )
    return float(out["minutes"])


def main() -> int:
    targets = parse_targets(TARGETS)
    inputs = make_inputs()
    py_url = PY_URL.rstrip("/")

    # 기준이 원본 모델이 아니면 비교 자체가 의미 없다
    try:
        model = get_json(f"{py_url}/health").get("model")
        expected = [predict(py_url, d, p) for d, p in inputs]
    except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
        print(f"FAIL [python] 기준 서버({py_url}) 호출 실패: {e}")
        return 1
    if model != BASE_MODEL:
        print(
            f"FAIL [python] 기준 서버 모델이 {model!r} 다 (MODEL_IMPL={BASE_MODEL} 로 띄울 것)"
        )
        return 1

    results = {}
    for name, url in targets.items():
        try:
            got = [predict(url, d, p) for d, p in inputs]
        except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
            print(f"FAIL [{name}] {url} 호출 실패: {e}")
            results[name] = False
            continue

        diffs = [abs(g - e) for g, e in zip(got, expected)]
        worst = max(range(len(diffs)), key=diffs.__getitem__)
        ok = diffs[worst] < TOLERANCE
        results[name] = ok
        d, p = inputs[worst]
        print(
            ("PASS " if ok else "FAIL ")
            + f"[{name}] 최대 오차 {diffs[worst]:.2e}분 {'<' if ok else '>='} {TOLERANCE:g}분"
            + f" (입력 distance_km={d:.4f}, pending_orders={p}:"
            + f" python={expected[worst]:.6f}, {name}={got[worst]:.6f})"
        )

    # 모든 대상이 PASS 면 종료코드 0, 아니면 1
    return 0 if results and all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
