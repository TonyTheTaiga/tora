from rust:1.88 as builder
workdir /usr/src/app
run rustup target add $(uname -m)-unknown-linux-musl
copy . .
run cargo build --release --target $(uname -m)-unknown-linux-musl
from scratch
arg TARGETARCH
copy --from=builder /usr/src/app/target/*-unknown-linux-musl/release/api /api
expose 8080
cmd ["/api"]
