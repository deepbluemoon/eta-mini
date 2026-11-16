# 저장소 루트를 빌드 컨텍스트로 쓴다:
#   docker build -f api-rust/Dockerfile -t delivery-eta-rust .
#   docker run --rm -p 8082:8080 delivery-eta-rust

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

FROM rust:1-bookworm AS build
WORKDIR /src/api-rust
# 의존성만 먼저 빌드해 소스가 바뀌어도 이 층을 재사용한다
COPY api-rust/Cargo.toml api-rust/Cargo.lock ./
RUN mkdir src && echo 'fn main() {}' > src/main.rs \
    && cargo build --release --locked && cargo test --release --locked --no-run \
    && rm -rf src
COPY api-rust/src ./src
# 테스트는 실제 모델과 런타임으로도 돌린다 (onnx_known_values)
COPY artifacts/model_sklearn.onnx /src/artifacts/model_sklearn.onnx
COPY --from=onnxruntime /opt/onnxruntime/lib/ /opt/onnxruntime/lib/
RUN touch src/main.rs \
    && ORT_LIB=/opt/onnxruntime/lib/libonnxruntime.so cargo test --release --locked \
    && cargo build --release --locked

FROM debian:bookworm-slim
COPY --from=onnxruntime /opt/onnxruntime/lib/ /usr/local/lib/
RUN ldconfig
WORKDIR /app
COPY --from=build /src/api-rust/target/release/api-rust /usr/local/bin/api-rust
COPY artifacts/model_sklearn.onnx artifacts/model_sklearn.onnx
ENV ORT_LIB=/usr/local/lib/libonnxruntime.so \
    MODEL_PATH=/app/artifacts/model_sklearn.onnx \
    PORT=8080
USER nobody
EXPOSE 8080
CMD ["api-rust"]
