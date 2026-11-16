package main

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"
)

// fakePredict 는 규칙 기반 공식(10 + 3*거리 + 2*주문)으로 ONNX Runtime 없이 계약만 확인한다.
func fakePredict(d, p float32) (float32, error) { return 10 + 3*d + 2*p, nil }

func do(t *testing.T, method, path, contentType, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	rec := httptest.NewRecorder()
	newMux(fakePredict).ServeHTTP(rec, req)
	return rec
}

// /health 가 200 과 status, model 을 돌려주는지 확인한다.
func TestHealth(t *testing.T) {
	rec := do(t, "GET", "/health", "", "")
	if rec.Code != 200 || strings.TrimSpace(rec.Body.String()) != `{"model":"onnx","status":"ok"}` {
		t.Fatalf("got %d %s", rec.Code, rec.Body)
	}
}

// 정상·경계 입력이 200 과 예상 minutes 를 돌려주는지 확인한다 (GET, POST 모두).
func TestEtaOK(t *testing.T) {
	cases := []struct {
		body string
		want float64
	}{
		{`{"distance_km": 3, "pending_orders": 2}`, 23},
		{`{"distance_km": 0, "pending_orders": 0}`, 10},
		{`{"distance_km": 1000, "pending_orders": 100000}`, 203010},
		{`{"distance_km": "2.5", "pending_orders": "2.0"}`, 21.5},
		{`{"distance_km": 1, "pending_orders": 2.0, "extra": 1}`, 17},
	}
	for _, method := range []string{"GET", "POST"} {
		for _, c := range cases {
			rec := do(t, method, "/eta", "application/json", c.body)
			var got etaResponse
			if rec.Code != 200 || json.Unmarshal(rec.Body.Bytes(), &got) != nil || got.Minutes != c.want || got.Model != "onnx" {
				t.Errorf("%s %s: got %d %s, want minutes=%v", method, c.body, rec.Code, rec.Body, c.want)
			}
		}
	}
}

// 잘못된 입력이 422 와 detail 목록을 돌려주는지 확인한다.
func TestEtaInvalid(t *testing.T) {
	cases := []struct{ contentType, body, wantType string }{
		{"application/json", `{"distance_km": -1, "pending_orders": 2}`, "greater_than_equal"},
		{"application/json", `{"distance_km": 1000.01, "pending_orders": 2}`, "less_than_equal"},
		{"application/json", `{"distance_km": 3, "pending_orders": -1}`, "greater_than_equal"},
		{"application/json", `{"distance_km": 3, "pending_orders": 2.5}`, "int_from_float"},
		{"application/json", `{"distance_km": "abc", "pending_orders": 2}`, "float_parsing"},
		{"application/json", `{"distance_km": "inf", "pending_orders": 2}`, "finite_number"},
		{"application/json", `{"distance_km": 1e400, "pending_orders": 2}`, "finite_number"},
		{"application/json", `{"distance_km": null, "pending_orders": 2}`, "float_parsing"},
		{"application/json", `{"pending_orders": 2}`, "missing"},
		{"application/json", `[]`, "model_attributes_type"},
		{"application/json", `nope`, "json_invalid"},
		{"application/json", ``, "missing"},
		{"text/plain", `{"distance_km": 3, "pending_orders": 2}`, "model_attributes_type"},
	}
	for _, c := range cases {
		rec := do(t, "GET", "/eta", c.contentType, c.body)
		var got struct{ Detail []fieldError }
		if rec.Code != 422 || json.Unmarshal(rec.Body.Bytes(), &got) != nil || len(got.Detail) == 0 || got.Detail[0].Type != c.wantType {
			t.Errorf("%q: got %d %s, want 422 %s", c.body, rec.Code, rec.Body, c.wantType)
		}
	}
}
