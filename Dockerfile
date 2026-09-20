# syntax=docker/dockerfile:1

# ---- build stage: static musl binary; the toolchain never reaches the final image
FROM rust:1-alpine AS build
# musl-dev + gcc: quantik-core bundles SQLite (rusqlite "bundled"), which compiles C.
# git: fetch quantik-core at its pinned revision (see below).
RUN apk add --no-cache musl-dev gcc git
WORKDIR /src

# Cargo.toml points quantik-core at a sibling checkout, which is not in the build
# context. Swap it for the immutable git revision the API records as CORE_REVISION
# (src/lib.rs) and documents in the README. Keep the two in sync.
ARG CORE_REV=2b35565dddc8e0f77222af2f8fcd382b013f2fee
COPY Cargo.toml Cargo.lock ./
RUN sed -i "s|^quantik-core = .*|quantik-core = { git = \"https://github.com/mberlanda/quantik-core-rust\", rev = \"${CORE_REV}\" }|" Cargo.toml \
    && grep -q "rev = \"${CORE_REV}\"" Cargo.toml
COPY src ./src
# No --locked: the lockfile's quantik-core entry is the path dependency and is
# re-resolved to the git revision; every other crate stays at its locked version.
RUN cargo build --release --bin quantik-api \
    && strip target/release/quantik-api

# ---- final stage: static binary on distroless, no shell, no toolchain, non-root
FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /src/target/release/quantik-api /usr/local/bin/quantik-api
# The binary defaults to 127.0.0.1:8000, unreachable from outside the container.
ENV QUANTIK_API_ADDR=0.0.0.0:8080
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/quantik-api"]
