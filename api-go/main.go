// delivery-eta API 의 Go 구현. api/app.py 와 같은 계약(GET /eta, GET /health)을 따른다.
//
// 예측은 ONNX 모델(입력 "X" [1, 2] float32 → 출력 "variable" [1, 1])을
// ONNX Runtime 으로 실행한다. 정규화는 모델 안에 있으므로 원래 단위의 값을 그대로 넣는다.
package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math"
	"mime"
	"net/http"
	"os"
	"strconv"
	"strings"

	ort "github.com/yalue/onnxruntime_go"
)

const (
	modelName     = "onnx" // /health, /eta 의 model 값 (장애 추적용)
	maxDistanceKm = 1000   // api/app.py 의 distance_km 상한과 맞춘다
)

// predictor 는 거리와 대기 주문 수로 예상 시간(분)을 계산한다. 테스트에서 바꿔 끼울 수 있게 함수로 둔다.
type predictor func(distanceKm float32, pendingOrders float32) (float32, error)

// fieldError 는 FastAPI 422 응답의 detail 항목과 같은 모양이다.
type fieldError struct {
	Type  string `json:"type"`
	Loc   []any  `json:"loc"`
	Msg   string `json:"msg"`
	Input any    `json:"input"`
}

type etaResponse struct {
	Minutes float64 `json:"minutes"`
	Model   string  `json:"model"`
}

func getenv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// newONNXPredictor 는 ONNX Runtime 세션을 만들고 예측 함수를 돌려준다.
// DynamicAdvancedSession 은 요청마다 텐서를 따로 넘기므로 동시에 불러도 안전하다.
func newONNXPredictor(libPath, modelPath string) (predictor, error) {
	ort.SetSharedLibraryPath(libPath)
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, fmt.Errorf("ONNX Runtime 초기화 실패(%s): %w", libPath, err)
	}
	session, err := ort.NewDynamicAdvancedSession(modelPath, []string{"X"}, []string{"variable"}, nil)
	if err != nil {
		return nil, fmt.Errorf("모델 로드 실패(%s): %w", modelPath, err)
	}
	predict := func(distanceKm, pendingOrders float32) (float32, error) {
		input, err := ort.NewTensor(ort.NewShape(1, 2), []float32{distanceKm, pendingOrders})
		if err != nil {
			return 0, err
		}
		defer input.Destroy()
		output, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 1))
		if err != nil {
			return 0, err
		}
		defer output.Destroy()
		if err := session.Run([]ort.Value{input}, []ort.Value{output}); err != nil {
			return 0, err
		}
		return output.GetData()[0], nil
	}
	return predict, nil
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	body, err := json.Marshal(v)
	if err != nil {
		// minutes 가 inf/NaN 이면 JSON 으로 못 만든다. Python API 도 이 경우 500 이다.
		http.Error(w, `{"detail":"Internal Server Error"}`, http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	w.Write(body)
}

func writeValidationError(w http.ResponseWriter, errs []fieldError) {
	writeJSON(w, http.StatusUnprocessableEntity, map[string]any{"detail": errs})
}

// parseNumber 는 pydantic 의 lax 모드처럼 JSON 숫자, 숫자 문자열, bool 을 float64 로 바꾼다.
func parseNumber(raw json.RawMessage) (float64, bool) {
	var v any
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	if err := dec.Decode(&v); err != nil {
		return 0, false
	}
	switch x := v.(type) {
	case json.Number:
		f, err := strconv.ParseFloat(x.String(), 64)
		return f, err == nil || errors.Is(err, strconv.ErrRange)
	case string:
		f, err := strconv.ParseFloat(strings.TrimSpace(x), 64)
		return f, err == nil || errors.Is(err, strconv.ErrRange)
	case bool:
		if x {
			return 1, true
		}
		return 0, true
	}
	return 0, false
}

// rawInput 은 422 응답의 input 에 그대로 실을 값이다.
func rawInput(raw json.RawMessage) any {
	var v any
	if json.Unmarshal(raw, &v) != nil {
		return string(raw)
	}
	return v
}

// validateEta 는 EtaRequest(api/app.py)와 같은 규칙으로 본문을 검사한다.
//   - distance_km: 0 이상 1000 이하의 유한한 수
//   - pending_orders: 0 이상의 정수
func validateEta(body []byte) (distanceKm, pendingOrders float64, errs []fieldError) {
	if len(bytes.TrimSpace(body)) == 0 {
		return 0, 0, []fieldError{{Type: "missing", Loc: []any{"body"}, Msg: "Field required", Input: nil}}
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(body, &obj); err != nil {
		var v any
		if json.Unmarshal(body, &v) == nil {
			// JSON 이지만 객체가 아니다 (배열, 숫자 등)
			return 0, 0, []fieldError{{Type: "model_attributes_type", Loc: []any{"body"},
				Msg: "Input should be a valid dictionary or object to extract fields from", Input: v}}
		}
		return 0, 0, []fieldError{{Type: "json_invalid", Loc: []any{"body", 0}, Msg: "JSON decode error", Input: map[string]any{}}}
	}

	loc := func(name string) []any { return []any{"body", name} }

	raw, ok := obj["distance_km"]
	switch {
	case !ok:
		errs = append(errs, fieldError{"missing", loc("distance_km"), "Field required", rawInput(body)})
	default:
		d, ok := parseNumber(raw)
		switch {
		case !ok:
			errs = append(errs, fieldError{"float_parsing", loc("distance_km"), "Input should be a valid number", rawInput(raw)})
		case math.IsNaN(d) || math.IsInf(d, 0):
			errs = append(errs, fieldError{"finite_number", loc("distance_km"), "Input should be a finite number", rawInput(raw)})
		case d < 0:
			errs = append(errs, fieldError{"greater_than_equal", loc("distance_km"), "Input should be greater than or equal to 0", rawInput(raw)})
		case d > maxDistanceKm:
			errs = append(errs, fieldError{"less_than_equal", loc("distance_km"), "Input should be less than or equal to 1000", rawInput(raw)})
		default:
			distanceKm = d
		}
	}

	raw, ok = obj["pending_orders"]
	switch {
	case !ok:
		errs = append(errs, fieldError{"missing", loc("pending_orders"), "Field required", rawInput(body)})
	default:
		p, ok := parseNumber(raw)
		switch {
		case !ok || math.IsNaN(p) || math.IsInf(p, 0):
			errs = append(errs, fieldError{"int_parsing", loc("pending_orders"), "Input should be a valid integer", rawInput(raw)})
		case p != math.Trunc(p):
			errs = append(errs, fieldError{"int_from_float", loc("pending_orders"), "Input should be a valid integer, got a number with a fractional part", rawInput(raw)})
		case p < 0:
			errs = append(errs, fieldError{"greater_than_equal", loc("pending_orders"), "Input should be greater than or equal to 0", rawInput(raw)})
		default:
			pendingOrders = p
		}
	}
	return distanceKm, pendingOrders, errs
}

// isJSON 은 Content-Type 이 JSON 인지 본다. FastAPI 는 그 밖의 본문을 JSON 으로 읽지 않는다.
func isJSON(contentType string) bool {
	mt, _, err := mime.ParseMediaType(contentType)
	if err != nil {
		return false
	}
	return mt == "application/json" || (strings.HasPrefix(mt, "application/") && strings.HasSuffix(mt, "+json"))
}

func newMux(predict predictor) *http.ServeMux {
	mux := http.NewServeMux()

	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok", "model": modelName})
	})

	eta := func(w http.ResponseWriter, r *http.Request) {
		body, err := readBody(r)
		if err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"detail": "본문을 읽을 수 없다"})
			return
		}
		if len(bytes.TrimSpace(body)) > 0 && !isJSON(r.Header.Get("Content-Type")) {
			writeValidationError(w, []fieldError{{Type: "model_attributes_type", Loc: []any{"body"},
				Msg: "Input should be a valid dictionary or object to extract fields from", Input: string(body)}})
			return
		}
		d, p, errs := validateEta(body)
		if len(errs) > 0 {
			writeValidationError(w, errs)
			return
		}
		minutes, err := predict(float32(d), float32(p))
		if err != nil {
			log.Printf("예측 실패: %v", err)
			writeJSON(w, http.StatusInternalServerError, map[string]string{"detail": "Internal Server Error"})
			return
		}
		writeJSON(w, http.StatusOK, etaResponse{Minutes: float64(minutes), Model: modelName})
	}
	// api/app.py 는 GET 에 JSON body 를 받는다. POST 도 같은 처리로 받는다.
	mux.HandleFunc("GET /eta", eta)
	mux.HandleFunc("POST /eta", eta)
	return mux
}

const maxBodyBytes = 1 << 20

func readBody(r *http.Request) ([]byte, error) {
	var buf bytes.Buffer
	_, err := buf.ReadFrom(http.MaxBytesReader(nil, r.Body, maxBodyBytes))
	return buf.Bytes(), err
}

func main() {
	libPath := getenv("ORT_LIB_PATH", "/usr/local/lib/libonnxruntime.so")
	modelPath := getenv("MODEL_PATH", "artifacts/model_sklearn.onnx")
	addr := ":" + getenv("PORT", "8080")

	predict, err := newONNXPredictor(libPath, modelPath)
	if err != nil {
		log.Fatal(err)
	}

	log.Printf("listening on %s (model=%s)", addr, modelPath)
	log.Fatal(http.ListenAndServe(addr, newMux(predict)))
}
