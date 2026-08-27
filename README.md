# quantik-api-rust

Standalone Axum gateway for the engines provided by `quantik-core-rust`.

The development dependency uses the sibling checkout. For an independently
deployed build, replace its `quantik-core` dependency with the immutable core
revision recorded by the API:

```toml
quantik-core = { git = "https://github.com/mberlanda/quantik-core-rust", rev = "2b35565dddc8e0f77222af2f8fcd382b013f2fee" }
```

## Run

```sh
cargo run --release
```

The default address is `127.0.0.1:8000`. Override it with
`QUANTIK_API_ADDR=0.0.0.0:9000`.

Endpoints:

- `GET /health`
- `GET /v1/engines`
- `POST /v1/move/minimax`
- `POST /v1/move/mcts`
- `POST /v1/move/beam`

Configure the visualizer with an endpoint such as:

```text
http://127.0.0.1:8000/v1/move/mcts
```

Requests use `quantik.engine-request.v1`, QFEN, side-to-move, and the complete
legal-action set. The service recalculates legality with `quantik-core` before
running an engine. Responses use `action_index = shape * 16 + position` and
include the pinned core revision as engine provenance.

Development CORS allows any origin. Restrict origins at the reverse proxy or in
the router before exposing this service publicly.
