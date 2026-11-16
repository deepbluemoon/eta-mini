//! delivery-eta API 의 Rust 구현. api/app.py 와 같은 계약(GET /eta, GET /health)을 따른다.
//!
//! 예측은 ONNX 모델(입력 "X" [1, 2] float32 → 출력 "variable" [1, 1])을
//! ONNX Runtime 으로 실행한다. 정규화는 모델 안에 있으므로 원래 단위의 값을 그대로 넣는다.

use std::sync::{Arc, Mutex};

use axum::{
    Json, Router,
    body::Bytes,
    extract::State,
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
    routing::get,
};
use ort::{session::Session, value::Tensor};
use serde::Serialize;
use serde_json::{Map, Value, json};

const MODEL_NAME: &str = "onnx"; // /health, /eta 의 model 값 (장애 추적용)
const MAX_DISTANCE_KM: f64 = 1000.0; // api/app.py 의 distance_km 상한과 맞춘다

/// 거리와 대기 주문 수로 예상 시간(분)을 계산한다. 테스트에서 바꿔 끼울 수 있게 함수로 둔다.
type Predictor = Arc<dyn Fn(f32, f32) -> Result<f32, String> + Send + Sync>;

/// FastAPI 422 응답의 detail 항목과 같은 모양이다.
#[derive(Serialize)]
struct FieldError {
    #[serde(rename = "type")]
    kind: &'static str,
    loc: Vec<Value>,
    msg: &'static str,
    input: Value,
}

impl FieldError {
    fn new(kind: &'static str, loc: Vec<Value>, msg: &'static str, input: Value) -> Self {
        Self { kind, loc, msg, input }
    }
}

#[derive(Serialize)]
struct EtaResponse {
    minutes: f64,
    model: &'static str,
}

fn getenv(key: &str, fallback: &str) -> String {
    std::env::var(key).ok().filter(|v| !v.is_empty()).unwrap_or_else(|| fallback.to_string())
}

/// ONNX Runtime 을 lib_path 에서 읽어 세션을 만들고 예측 함수를 돌려준다.
/// Session::run 이 &mut self 를 받으므로 Mutex 로 감싼다.
fn new_onnx_predictor(lib_path: &str, model_path: &str) -> Result<Predictor, String> {
    let env = ort::init_from(lib_path)
        .map_err(|e| format!("ONNX Runtime 로드 실패({lib_path}): {e}"))?
        .build()
        .map_err(|e| format!("ONNX Runtime 초기화 실패({lib_path}): {e}"))?;
    let session = Session::builder(&env)
        .and_then(|mut b| b.commit_from_file(model_path))
        .map_err(|e| format!("모델 로드 실패({model_path}): {e}"))?;
    // 환경은 프로세스가 끝날 때까지 살아 있어야 한다
    std::mem::forget(env);
    let session = Mutex::new(session);

    Ok(Arc::new(move |distance_km, pending_orders| {
        let input = Tensor::from_array(([1usize, 2], vec![distance_km, pending_orders]))
            .map_err(|e| e.to_string())?;
        let mut session = session.lock().map_err(|e| e.to_string())?;
        let outputs = session.run(ort::inputs!["X" => input]).map_err(|e| e.to_string())?;
        let (_, data) = outputs["variable"]
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?;
        data.first().copied().ok_or_else(|| "빈 출력".to_string())
    }))
}

fn validation_error(errs: Vec<FieldError>) -> Response {
    (StatusCode::UNPROCESSABLE_ENTITY, Json(json!({ "detail": errs }))).into_response()
}

fn internal_error() -> Response {
    (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "detail": "Internal Server Error" })))
        .into_response()
}

/// pydantic 의 lax 모드처럼 JSON 숫자, 숫자 문자열, bool 을 f64 로 바꾼다.
/// 범위를 넘는 수(1e400 등)는 inf 가 되어 뒤에서 finite_number 로 거절된다.
fn parse_number(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.to_string().parse().ok(),
        Value::String(s) => s.trim().parse().ok(),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

/// EtaRequest(api/app.py)와 같은 규칙으로 본문을 검사한다.
///   - distance_km: 0 이상 1000 이하의 유한한 수
///   - pending_orders: 0 이상의 정수
fn validate_eta(body: &[u8]) -> Result<(f64, f64), Vec<FieldError>> {
    if body.trim_ascii().is_empty() {
        return Err(vec![FieldError::new("missing", vec![json!("body")], "Field required", Value::Null)]);
    }
    let obj: Map<String, Value> = match serde_json::from_slice::<Value>(body) {
        Ok(Value::Object(obj)) => obj,
        // JSON 이지만 객체가 아니다 (배열, 숫자 등)
        Ok(v) => {
            return Err(vec![FieldError::new(
                "model_attributes_type",
                vec![json!("body")],
                "Input should be a valid dictionary or object to extract fields from",
                v,
            )]);
        }
        Err(_) => {
            return Err(vec![FieldError::new(
                "json_invalid",
                vec![json!("body"), json!(0)],
                "JSON decode error",
                json!({}),
            )]);
        }
    };

    let loc = |name: &str| vec![json!("body"), json!(name)];
    let whole = || Value::Object(obj.clone());
    let mut errs = Vec::new();
    let (mut distance_km, mut pending_orders) = (0.0, 0.0);

    match obj.get("distance_km") {
        None => errs.push(FieldError::new("missing", loc("distance_km"), "Field required", whole())),
        Some(raw) => match parse_number(raw) {
            None => errs.push(FieldError::new(
                "float_parsing",
                loc("distance_km"),
                "Input should be a valid number",
                raw.clone(),
            )),
            Some(d) if !d.is_finite() => errs.push(FieldError::new(
                "finite_number",
                loc("distance_km"),
                "Input should be a finite number",
                raw.clone(),
            )),
            Some(d) if d < 0.0 => errs.push(FieldError::new(
                "greater_than_equal",
                loc("distance_km"),
                "Input should be greater than or equal to 0",
                raw.clone(),
            )),
            Some(d) if d > MAX_DISTANCE_KM => errs.push(FieldError::new(
                "less_than_equal",
                loc("distance_km"),
                "Input should be less than or equal to 1000",
                raw.clone(),
            )),
            Some(d) => distance_km = d,
        },
    }

    match obj.get("pending_orders") {
        None => errs.push(FieldError::new("missing", loc("pending_orders"), "Field required", whole())),
        Some(raw) => match parse_number(raw) {
            Some(p) if p.is_finite() && p != p.trunc() => errs.push(FieldError::new(
                "int_from_float",
                loc("pending_orders"),
                "Input should be a valid integer, got a number with a fractional part",
                raw.clone(),
            )),
            Some(p) if p.is_finite() && p < 0.0 => errs.push(FieldError::new(
                "greater_than_equal",
                loc("pending_orders"),
                "Input should be greater than or equal to 0",
                raw.clone(),
            )),
            Some(p) if p.is_finite() => pending_orders = p,
            _ => errs.push(FieldError::new(
                "int_parsing",
                loc("pending_orders"),
                "Input should be a valid integer",
                raw.clone(),
            )),
        },
    }

    if errs.is_empty() { Ok((distance_km, pending_orders)) } else { Err(errs) }
}

/// Content-Type 이 JSON 인지 본다. FastAPI 는 그 밖의 본문을 JSON 으로 읽지 않는다.
fn is_json(headers: &HeaderMap) -> bool {
    let Some(mt) = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<mime::Mime>().ok())
    else {
        return false;
    };
    mt.type_() == mime::APPLICATION
        && (mt.subtype() == mime::JSON || mt.suffix() == Some(mime::JSON))
}

async fn health() -> Json<Value> {
    Json(json!({ "status": "ok", "model": MODEL_NAME }))
}

// axum 의 Json 추출기는 형식 오류에 400·415 를 돌려주므로 본문을 직접 읽어 모든 입력 오류를 422 로 맞춘다
async fn eta(State(predict): State<Predictor>, headers: HeaderMap, body: Bytes) -> Response {
    if !body.trim_ascii().is_empty() && !is_json(&headers) {
        let input = Value::String(String::from_utf8_lossy(&body).into_owned());
        return validation_error(vec![FieldError::new(
            "model_attributes_type",
            vec![json!("body")],
            "Input should be a valid dictionary or object to extract fields from",
            input,
        )]);
    }
    let (d, p) = match validate_eta(&body) {
        Ok(v) => v,
        Err(errs) => return validation_error(errs),
    };
    let minutes = match tokio::task::spawn_blocking(move || predict(d as f32, p as f32)).await {
        Ok(Ok(m)) => m as f64,
        Ok(Err(e)) => {
            eprintln!("예측 실패: {e}");
            return internal_error();
        }
        Err(e) => {
            eprintln!("예측 작업 실패: {e}");
            return internal_error();
        }
    };
    // minutes 가 inf/NaN 이면 JSON 으로 못 만든다. Python API 도 이 경우 500 이다.
    if !minutes.is_finite() {
        return internal_error();
    }
    Json(EtaResponse { minutes, model: MODEL_NAME }).into_response()
}

fn app(predict: Predictor) -> Router {
    // api/app.py 는 GET 에 JSON body 를 받는다. POST 도 같은 처리로 받는다.
    Router::new()
        .route("/health", get(health))
        .route("/eta", get(eta).post(eta))
        .with_state(predict)
}

#[tokio::main]
async fn main() {
    let lib_path = getenv("ORT_LIB", "/usr/local/lib/libonnxruntime.so");
    let model_path = getenv("MODEL_PATH", "artifacts/model_sklearn.onnx");
    let addr = format!("0.0.0.0:{}", getenv("PORT", "8080"));

    let predict = new_onnx_predictor(&lib_path, &model_path).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap_or_else(|e| {
        eprintln!("{addr} 에 바인드 실패: {e}");
        std::process::exit(1);
    });
    eprintln!("listening on {addr} (model={model_path})");
    axum::serve(listener, app(predict))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
        .unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, http::Request};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    /// 규칙 기반 공식(10 + 3*거리 + 2*주문)으로 ONNX Runtime 없이 계약만 확인한다.
    fn fake_predict() -> Predictor {
        Arc::new(|d, p| Ok(10.0 + 3.0 * d + 2.0 * p))
    }

    async fn call(method: &str, path: &str, content_type: &str, body: &str) -> (StatusCode, Value) {
        let mut req = Request::builder().method(method).uri(path);
        if !content_type.is_empty() {
            req = req.header(header::CONTENT_TYPE, content_type);
        }
        let resp = app(fake_predict())
            .oneshot(req.body(Body::from(body.to_string())).unwrap())
            .await
            .unwrap();
        let status = resp.status();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
    }

    /// /health 가 200 과 status, model 을 돌려주는지 확인한다.
    #[tokio::test]
    async fn health_ok() {
        let (status, body) = call("GET", "/health", "", "").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, json!({ "status": "ok", "model": "onnx" }));
    }

    /// 정상·경계 입력이 200 과 예상 minutes 를 돌려주는지 확인한다 (GET, POST 모두).
    #[tokio::test]
    async fn eta_ok() {
        let cases = [
            (r#"{"distance_km": 3, "pending_orders": 2}"#, 23.0),
            (r#"{"distance_km": 0, "pending_orders": 0}"#, 10.0),
            (r#"{"distance_km": 1000, "pending_orders": 100000}"#, 203010.0),
            (r#"{"distance_km": "2.5", "pending_orders": "2.0"}"#, 21.5),
            (r#"{"distance_km": 1, "pending_orders": 2.0, "extra": 1}"#, 17.0),
        ];
        for method in ["GET", "POST"] {
            for (body, want) in cases {
                let (status, got) = call(method, "/eta", "application/json", body).await;
                assert_eq!(status, StatusCode::OK, "{method} {body}: {got}");
                assert_eq!(got, json!({ "minutes": want, "model": "onnx" }), "{method} {body}");
            }
        }
    }

    /// 잘못된 입력이 422 와 detail 목록을 돌려주는지 확인한다 (400·415 가 아님).
    #[tokio::test]
    async fn eta_invalid() {
        let cases = [
            ("application/json", r#"{"distance_km": -1, "pending_orders": 2}"#, "greater_than_equal"),
            ("application/json", r#"{"distance_km": 1000.01, "pending_orders": 2}"#, "less_than_equal"),
            ("application/json", r#"{"distance_km": 3, "pending_orders": -1}"#, "greater_than_equal"),
            ("application/json", r#"{"distance_km": 3, "pending_orders": 2.5}"#, "int_from_float"),
            ("application/json", r#"{"distance_km": "abc", "pending_orders": 2}"#, "float_parsing"),
            ("application/json", r#"{"distance_km": "inf", "pending_orders": 2}"#, "finite_number"),
            ("application/json", r#"{"distance_km": 1e400, "pending_orders": 2}"#, "finite_number"),
            ("application/json", r#"{"distance_km": null, "pending_orders": 2}"#, "float_parsing"),
            ("application/json", r#"{"pending_orders": 2}"#, "missing"),
            ("application/json", "[]", "model_attributes_type"),
            ("application/json", "nope", "json_invalid"),
            ("application/json", "", "missing"),
            ("", r#"{"distance_km": 3, "pending_orders": 2}"#, "model_attributes_type"),
            ("text/plain", r#"{"distance_km": 3, "pending_orders": 2}"#, "model_attributes_type"),
        ];
        for (content_type, body, want) in cases {
            let (status, got) = call("GET", "/eta", content_type, body).await;
            assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body:?}: {got}");
            assert_eq!(got["detail"][0]["type"], want, "{body:?}: {got}");
        }
    }

    /// 실제 ONNX 모델이 Python(onnxruntime)과 같은 값을 내는지 확인한다. ORT_LIB 이 없으면 건너뛴다.
    #[test]
    fn onnx_known_values() {
        let Ok(lib) = std::env::var("ORT_LIB") else {
            eprintln!("ORT_LIB 이 없어 건너뛴다");
            return;
        };
        let model = concat!(env!("CARGO_MANIFEST_DIR"), "/../artifacts/model_sklearn.onnx");
        let predict = new_onnx_predictor(&lib, model).unwrap();
        // CI 는 모델을 다시 학습하고 CPU 도 달라 f32 끝자리가 흔들린다. check_parity 와 같은 허용 오차로 비교한다
        for ((d, p), want) in [((3.0, 2.0), 28.940723419189453), ((0.0, 0.0), 12.031005859375)] {
            let got = predict(d, p).unwrap();
            assert!((got - want).abs() < 1e-4, "predict({d}, {p}) = {got}, want {want}");
        }
    }
}
