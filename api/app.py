import importlib
import os
from string import Template

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

APP_VERSION = "0.2.0"  # pyproject.toml 의 version 과 맞춘다
MODEL_IMPL = os.getenv("MODEL_IMPL","rule")
estimate_delivery_minutes = importlib.import_module(f"api.predictors.{MODEL_IMPL}").estimate_delivery_minutes

class EtaRequest(BaseModel):
    """배달 예상 시간 요청"""
    # 모델이 더 많은 정보를 원하면 여기에 필드를 추가
    # 아래 ge=09 와 같은 조건들을 유효성 검사 룰을 지정 (validation rule)
    distance_km: float = Field(ge=0, description="가게에서 배달지까지 거리(km)")
    pending_orders: int = Field(ge=0, description="지금 밀려 있는 주문 수")

class EtaResponse(BaseModel):
    """배달 예상 시간 응답"""
    minutes: float
    model: str # 어떤 구현체가 낸 예측인지 (장애 추적용)

app = FastAPI(title="delivery-eta", version=APP_VERSION)

# CSS 의 { } 와 섞이지 않도록 f-string 대신 $name 치환을 쓴다
INDEX_HTML = Template("""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>delivery-eta API</title>
  <style>
    :root { color-scheme: light dark; --fg: #1f2328; --muted: #59636e; --bg: #ffffff; --card: #f6f8fa; --line: #d1d9e0; --accent: #0969da; }
    @media (prefers-color-scheme: dark) {
      :root { --fg: #e6edf3; --muted: #9198a1; --bg: #0d1117; --card: #161b22; --line: #3d444d; --accent: #4493f8; }
    }
    body { margin: 0; background: var(--bg); color: var(--fg); font: 16px/1.6 system-ui, -apple-system, "Apple SD Gothic Neo", sans-serif; }
    main { max-width: 760px; margin: 0 auto; padding: 40px 16px; }
    h1 { margin: 0 0 4px; font-size: 1.8rem; }
    .meta { color: var(--muted); margin: 0 0 32px; }
    .badge { display: inline-block; padding: 0 8px; border: 1px solid var(--line); border-radius: 999px; font-size: .85rem; }
    h2 { font-size: 1.15rem; margin: 32px 0 12px; }
    table { width: 100%; border-collapse: collapse; }
    th, td { text-align: left; padding: 8px; border-bottom: 1px solid var(--line); vertical-align: top; }
    code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .9rem; }
    pre { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 12px; overflow-x: auto; }
    a { color: var(--accent); }
  </style>
</head>
<body>
<main>
  <h1>delivery-eta API</h1>
  <p class="meta">배달 예상 시간(분)을 계산하는 API
    &middot; <span class="badge">v$version</span>
    &middot; <span class="badge">model: $model</span></p>

  <h2>엔드포인트</h2>
  <table>
    <tr><th>메서드</th><th>경로</th><th>설명</th></tr>
    <tr><td><code>GET</code></td><td><code>/</code></td><td>이 소개 페이지</td></tr>
    <tr><td><code>GET</code></td><td><code>/health</code></td><td>서비스 상태와 현재 구현체 이름</td></tr>
    <tr><td><code>GET</code></td><td><code>/eta</code></td><td>JSON body 로 <code>distance_km</code>(km, 0 이상)와 <code>pending_orders</code>(건, 0 이상)를 받아 예상 시간(분)을 돌려준다</td></tr>
  </table>

  <h2>호출 예시</h2>
  <pre>curl -X GET http://localhost:8080/eta \\
  -H 'Content-Type: application/json' \\
  -d '{"distance_km": 2.5, "pending_orders": 3}'

{"minutes": 23.5, "model": "rule"}</pre>

  <h2>문서</h2>
  <p><a href="/docs">Swagger UI</a> &middot; <a href="/redoc">ReDoc</a> &middot; <a href="/openapi.json">OpenAPI 스키마</a></p>
</main>
</body>
</html>
""")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """API 소개 페이지(HTML)를 돌려준다."""
    return INDEX_HTML.substitute(version=APP_VERSION, model=MODEL_IMPL)


@app.get("/health")
def health() -> dict[str, str]:
    """서비스 상태와 현재 구현체 이름을 돌려준다.
    """
    return {"status": "ok","model": MODEL_IMPL}

@app.get("/eta", response_model=EtaResponse)
def eta(req: EtaRequest) -> EtaResponse:
    """배달 예상 시간을 계산"""
    minutes = estimate_delivery_minutes(req.distance_km, req.pending_orders)
    return EtaResponse(minutes=minutes, model=MODEL_IMPL)
