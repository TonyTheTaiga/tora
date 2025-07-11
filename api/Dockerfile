FROM rust:1.88-alpine AS builder
RUN apk add --no-cache musl-dev openssl-dev openssl-libs-static pkgconfig
WORKDIR /app
COPY Cargo.toml ./
RUN cargo generate-lockfile 2>/dev/null || true
COPY src/ ./src/
RUN cargo build --release

FROM alpine:latest
RUN apk add --no-cache ca-certificates
WORKDIR /app
COPY --from=builder /app/target/release/api ./api
RUN addgroup -g 1001 -S appgroup &&     adduser -S appuser -u 1001 -G appgroup
RUN chown -R appuser:appgroup /app
USER appuser
EXPOSE 8080
ENV RUST_LOG=info
ENV RUST_BACKTRACE=1
ENV RUST_ENV=production
CMD ["./api"]
STOPSIGNAL SIGTERM
