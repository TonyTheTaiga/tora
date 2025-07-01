FROM rust:1.88-alpine AS api-builder
RUN apk add --no-cache musl-dev openssl-dev
WORKDIR /app/api
COPY api/Cargo.toml ./
RUN cargo generate-lockfile 2>/dev/null || true
COPY api/src/ ./src/
RUN cargo build --release

FROM node:22-alpine AS frontend-builder
RUN npm install -g pnpm
WORKDIR /app/web-new
COPY web-new/package.json web-new/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
COPY web-new/ .
RUN pnpm run build:production
FROM alpine:latest AS runtime
RUN apk add --no-cache ca-certificates
WORKDIR /app

COPY --from=api-builder /app/api/target/release/api ./api
COPY --from=frontend-builder /app/web-new/build ./static
RUN addgroup -g 1001 -S appgroup && \
    adduser -S appuser -u 1001 -G appgroup
RUN chown -R appuser:appgroup /app
USER appuser
EXPOSE 8080
ENV RUST_LOG=info
ENV RUST_BACKTRACE=1
ENV STATIC_FILES_PATH=./static
CMD ["./api"]
STOPSIGNAL SIGTERM
