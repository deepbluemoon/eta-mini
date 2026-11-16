# 저장소 루트를 빌드 컨텍스트로 쓴다:
#   docker build -f api-go/Dockerfile -t delivery-eta-go .
#   docker run --rm -p 8081:8080 delivery-eta-go

# ONNX Runtime 버전은 uv.lock 에서 읽는다. Python API 와 같은 런타임으로 맞추기 위해서다.
FROM debian:bookworm-slim AS onnxruntime
ARG TARGETARCH
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
COPY uv.lock /tmp/uv.lock
RUN set -eu; \
    ORT_VERSION="$(awk '/^name = "onnxruntime"$/ { getline; gsub(/version = |"/, ""); print; exit }' /tmp/uv.lock)"; \
    test -n "$ORT_VERSION" || { echo "uv.lock 에서 onnxruntime 버전을 찾지 못했다" >&2; exit 1; }; \
    case "$TARGETARCH" in \
      amd64) ORT_ARCH=x64 ;; \
      arm64) ORT_ARCH=aarch64 ;; \
      *) echo "지원하지 않는 TARGETARCH: $TARGETARCH" >&2; exit 1 ;; \
    esac; \
    curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-${ORT_ARCH}-${ORT_VERSION}.tgz" \
      | tar -xz -C /opt; \
    mv "/opt/onnxruntime-linux-${ORT_ARCH}-${ORT_VERSION}" /opt/onnxruntime

FROM golang:1.27-bookworm AS build
WORKDIR /src
COPY api-go/go.mod api-go/go.sum ./
RUN go mod download
COPY api-go/ ./
RUN go test ./... && CGO_ENABLED=1 go build -trimpath -ldflags="-s -w" -o /out/api-go .

FROM debian:bookworm-slim
COPY --from=onnxruntime /opt/onnxruntime/lib/ /usr/local/lib/
RUN ldconfig
WORKDIR /app
COPY --from=build /out/api-go /usr/local/bin/api-go
COPY artifacts/model_sklearn.onnx artifacts/model_sklearn.onnx
ENV ORT_LIB_PATH=/usr/local/lib/libonnxruntime.so \
    MODEL_PATH=/app/artifacts/model_sklearn.onnx \
    PORT=8080
USER nobody
EXPOSE 8080
CMD ["api-go"]
