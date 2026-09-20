use std::time::Instant;

use axum::{
    extract::Path,
    http::{header, Method, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use quantik_core::{
    beam_search::{BeamSearchConfig, BeamSearchEngine},
    mcts::{MCTSConfig, MCTSEngine},
    minimax::{MinimaxConfig, MinimaxEngine},
    moves::{apply_move, generate_legal_moves, Move},
    state::State,
};
use serde::{Deserialize, Serialize};
use tower_http::{cors::CorsLayer, trace::TraceLayer};

pub const API_VERSION: &str = env!("CARGO_PKG_VERSION");
pub const CORE_REVISION: &str = "2b35565dddc8e0f77222af2f8fcd382b013f2fee";
pub const REQUEST_SCHEMA: &str = "engine-request.v1";
pub const RESPONSE_SCHEMA: &str = "engine-response.v2";
/// Longest candidate list a response carries (QW-018 W1, section 1).
pub const MAX_CANDIDATES: usize = 8;
/// Pre-registration spelling of the request schema, still accepted on input
/// (QW-019 decisions.md#D3). Never emitted. Removed at the next minor release:
/// deleting this const, its branch in `validate_request`, and its test is the
/// whole change.
pub const REQUEST_SCHEMA_LEGACY: &str = "quantik.engine-request.v1";

pub fn app() -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/engines", get(list_engines))
        .route("/v1/move/{engine}", post(choose_move))
        .layer(
            CorsLayer::new()
                .allow_origin(tower_http::cors::Any)
                .allow_methods([Method::GET, Method::POST])
                .allow_headers([header::CONTENT_TYPE]),
        )
        .layer(TraceLayer::new_for_http())
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
    service: &'static str,
    version: &'static str,
    core_revision: &'static str,
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        service: "quantik-api",
        version: API_VERSION,
        core_revision: CORE_REVISION,
    })
}

#[derive(Debug, Serialize)]
struct EngineDescriptor {
    kind: &'static str,
    core_revision: &'static str,
}

async fn list_engines() -> Json<Vec<EngineDescriptor>> {
    Json(
        ["minimax", "mcts", "beam"]
            .into_iter()
            .map(|kind| EngineDescriptor {
                kind,
                core_revision: CORE_REVISION,
            })
            .collect(),
    )
}

#[derive(Clone, Debug, Deserialize)]
pub struct MoveRequest {
    pub schema: String,
    pub qfen: String,
    pub side_to_move: u8,
    pub legal_action_indices: Vec<u8>,
    #[serde(default)]
    pub config: SearchConfig,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct SearchConfig {
    pub max_depth: Option<u32>,
    pub time_limit_ms: Option<u64>,
    pub iterations: Option<u32>,
    pub beam_width: Option<usize>,
    pub rollouts: Option<u32>,
    pub seed: Option<u64>,
}

/// Whether the numbers in a response are proven or guessed. Required on the
/// wire (`engine-response.v2`); deliberately no `Default`, so every response
/// path has to say which it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Certainty {
    Estimate,
    Proof,
}

/// A candidate's score in the engine's own unit; the unit says which.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub enum Score {
    Count(u64),
    Value(f64),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Unit {
    Visits,
    Value,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Candidate {
    pub action_index: u8,
    pub score: Score,
    pub unit: Unit,
}

#[derive(Debug, Serialize)]
pub struct MoveResponse {
    pub schema: &'static str,
    pub action_index: u8,
    pub engine_kind: String,
    pub engine_version: &'static str,
    pub elapsed_ms: u64,
    pub certainty: Certainty,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidates: Option<Vec<Candidate>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pv: Option<Vec<u8>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub engine_config: Option<String>,
}

/// What one engine run produced, before it is checked against the position.
struct Outcome {
    best_move: Move,
    value: Option<f64>,
    certainty: Certainty,
    /// Engine-ranked, best first. Sanitised by `run_search`.
    candidates: Vec<(Move, Score, Unit)>,
    pv: Vec<Move>,
    config: String,
}

async fn choose_move(
    Path(engine): Path<String>,
    Json(request): Json<MoveRequest>,
) -> Result<Json<MoveResponse>, ApiError> {
    validate_request(&request)?;
    let engine_for_search = engine.clone();
    let result = tokio::task::spawn_blocking(move || run_search(&engine_for_search, request))
        .await
        .map_err(|error| ApiError::internal(format!("engine task failed: {error}")))??;
    Ok(Json(result))
}

fn validate_request(request: &MoveRequest) -> Result<(), ApiError> {
    if request.schema != REQUEST_SCHEMA && request.schema != REQUEST_SCHEMA_LEGACY {
        return Err(ApiError::bad_request(format!(
            "schema must be {REQUEST_SCHEMA}"
        )));
    }
    if request.side_to_move > 1 {
        return Err(ApiError::bad_request("side_to_move must be 0 or 1"));
    }
    if request.legal_action_indices.iter().any(|index| *index > 63) {
        return Err(ApiError::bad_request(
            "legal_action_indices must contain values from 0 through 63",
        ));
    }
    Ok(())
}

fn run_search(engine: &str, request: MoveRequest) -> Result<MoveResponse, ApiError> {
    let state = State::from_qfen(&request.qfen).map_err(ApiError::bad_request)?;
    let legal_moves = generate_legal_moves(&state.bb);
    let current_player = legal_moves
        .first()
        .map(|mv| mv.player)
        .ok_or_else(|| ApiError::unprocessable("position is terminal or has no legal moves"))?;
    if current_player != request.side_to_move {
        return Err(ApiError::unprocessable(format!(
            "side_to_move is {}, but core calculated {current_player}",
            request.side_to_move
        )));
    }

    let core_actions: Vec<u8> = legal_moves.iter().map(action_index).collect();
    let mut requested_actions = request.legal_action_indices.clone();
    requested_actions.sort_unstable();
    requested_actions.dedup();
    if core_actions != requested_actions {
        return Err(ApiError::unprocessable(
            "legal_action_indices do not exactly match quantik-core",
        ));
    }

    let started = Instant::now();
    let outcome = match engine {
        "minimax" => search_minimax(&state, &request.config)?,
        "mcts" => search_mcts(&state, &request.config)?,
        "beam" => search_beam(&state, &request.config)?,
        _ => return Err(ApiError::not_found(format!("unknown engine {engine:?}"))),
    };
    let selected_action = action_index(&outcome.best_move);
    if !core_actions.contains(&selected_action) {
        return Err(ApiError::internal(
            "engine returned an action that quantik-core considers illegal",
        ));
    }

    Ok(MoveResponse {
        schema: RESPONSE_SCHEMA,
        action_index: selected_action,
        engine_kind: engine.to_owned(),
        engine_version: CORE_REVISION,
        elapsed_ms: started.elapsed().as_millis().min(u64::MAX as u128) as u64,
        certainty: outcome.certainty,
        value: outcome.value,
        candidates: sanitise_candidates(&outcome.candidates, &core_actions, selected_action),
        pv: replayable_pv(&state, &outcome.pv, selected_action),
        engine_config: Some(outcome.config),
    })
}

/// Keep only legal, distinct candidates, the selected move first, at most
/// `MAX_CANDIDATES`. The engine's own order is otherwise preserved. A list
/// that does not contain the selected move is dropped rather than sent: the
/// selected move must be rank 0.
fn sanitise_candidates(
    ranked: &[(Move, Score, Unit)],
    legal: &[u8],
    selected: u8,
) -> Option<Vec<Candidate>> {
    let mut list: Vec<Candidate> = Vec::new();
    for (mv, score, unit) in ranked {
        let index = action_index(mv);
        if legal.contains(&index) && !list.iter().any(|c| c.action_index == index) {
            list.push(Candidate {
                action_index: index,
                score: *score,
                unit: *unit,
            });
        }
    }
    let position = list.iter().position(|c| c.action_index == selected)?;
    let chosen = list.remove(position);
    list.insert(0, chosen);
    list.truncate(MAX_CANDIDATES);
    Some(list)
}

/// The PV as action indices, only if it starts at the selected move and
/// replays legally from the root. Never empty: an unusable line is omitted.
fn replayable_pv(state: &State, pv: &[Move], selected: u8) -> Option<Vec<u8>> {
    if pv.first().map(action_index) != Some(selected) || pv.len() > 64 {
        return None;
    }
    let mut bb = state.bb;
    for mv in pv {
        if !generate_legal_moves(&bb).contains(mv) {
            return None;
        }
        bb = apply_move(&bb, mv);
    }
    Some(pv.iter().map(action_index).collect())
}

/// A mover-relative value of exactly +-1 is the core's proven-exclusive range
/// (`search_telemetry`: mate scores map to +-1, heuristics stay inside).
fn is_proven(value: f64) -> bool {
    value.abs() == 1.0
}

fn search_minimax(state: &State, input: &SearchConfig) -> Result<Outcome, ApiError> {
    let config = MinimaxConfig {
        max_depth: input.max_depth.unwrap_or(6).clamp(1, 16),
        time_limit_s: seconds(input.time_limit_ms),
        random_seed: input.seed,
        ..MinimaxConfig::default()
    };
    let max_depth = config.max_depth;
    let mut engine = MinimaxEngine::new(config);
    let result = engine.search(state).map_err(ApiError::unprocessable)?;
    let telemetry = engine.telemetry();
    let value = telemetry.as_ref().map(|t| t.root_value);
    // Per-root-move values from the core's public telemetry. With the default
    // `dedup_children` symmetric root moves are collapsed onto one
    // representative, so this is a ranked subset of the legal moves, which is
    // all the contract promises. The selected move is always in it.
    let mut scored: Vec<(Move, f64)> = telemetry
        .map(|t| {
            t.root_moves
                .iter()
                .filter_map(|r| r.q_value.map(|q| (r.mv, q)))
                .collect()
        })
        .unwrap_or_default();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));

    // `proof` iff every number sent is proven (W1 section 3, option A): a
    // depth-16 search always ends on true terminals; below that, a score is
    // proven only in the mate range. When the headline is proven the heuristic
    // tail is left out, so the response can honestly claim `proof`.
    let solved = result.depth_reached >= 16 && max_depth >= 16;
    let headline_proven = solved || value.is_some_and(is_proven);
    if headline_proven && !solved {
        scored.retain(|(_, q)| is_proven(*q));
    }
    let certainty = if headline_proven {
        Certainty::Proof
    } else {
        Certainty::Estimate
    };
    Ok(Outcome {
        best_move: result.best_move,
        value,
        certainty,
        candidates: scored
            .into_iter()
            .map(|(mv, q)| (mv, Score::Value(q), Unit::Value))
            .collect(),
        pv: result.pv,
        config: format!("depth={max_depth}"),
    })
}

fn search_mcts(state: &State, input: &SearchConfig) -> Result<Outcome, ApiError> {
    let config = MCTSConfig {
        max_iterations: input.iterations.unwrap_or(1_500).clamp(1, 1_000_000),
        max_depth: input.max_depth.unwrap_or(16).clamp(1, 16),
        time_limit_s: seconds(input.time_limit_ms),
        seed: input.seed,
        ..MCTSConfig::default()
    };
    let iterations = config.max_iterations;
    let mut engine = MCTSEngine::new(config);
    let (best_move, win_probability) = engine
        .search(&state.bb)
        .ok_or_else(|| ApiError::unprocessable("MCTS found no move"))?;
    // The default transposition table merges symmetric root moves, so this is
    // a ranked subset of the legal moves, not all of them.
    let mut visits = engine.root_move_visits();
    visits.sort_by_key(|entry| std::cmp::Reverse(entry.1));
    Ok(Outcome {
        best_move,
        value: Some(2.0 * win_probability - 1.0),
        certainty: Certainty::Estimate,
        candidates: visits
            .into_iter()
            .map(|(mv, n)| (mv, Score::Count(u64::from(n)), Unit::Visits))
            .collect(),
        pv: Vec::new(),
        config: format!("iterations={iterations}"),
    })
}

fn search_beam(state: &State, input: &SearchConfig) -> Result<Outcome, ApiError> {
    let config = BeamSearchConfig {
        beam_width: input.beam_width.unwrap_or(64).clamp(1, 100_000),
        max_depth: input.max_depth.unwrap_or(8).clamp(1, 16),
        rollouts_per_candidate: input.rollouts.unwrap_or(8).clamp(1, 100_000),
        random_seed: input.seed,
        time_limit_s: seconds(input.time_limit_ms),
        ..BeamSearchConfig::default()
    };
    let beam_width = config.beam_width;
    let result = BeamSearchEngine::new(config)
        .map_err(ApiError::bad_request)?
        .search(&state.bb)
        .map_err(ApiError::unprocessable)?;
    let ranked = result.ranked_root_moves(None);
    let candidates = ranked
        .iter()
        .map(|r| {
            (
                r.mv,
                Score::Value(r.best_value.clamp(-1.0, 1.0)),
                Unit::Value,
            )
        })
        .collect();
    let config = format!("beam_width={beam_width}");
    if let Some(leaf) = result
        .best_leaf
        .as_ref()
        .filter(|leaf| !leaf.moves.is_empty())
    {
        let root_value = if result.root_player == 0 {
            leaf.value
        } else {
            -leaf.value
        };
        return Ok(Outcome {
            best_move: leaf.moves[0],
            value: Some(root_value),
            certainty: Certainty::Estimate,
            candidates,
            pv: leaf.moves.clone(),
            config,
        });
    }
    ranked
        .first()
        .map(|first| Outcome {
            best_move: first.mv,
            value: Some(first.best_value),
            certainty: Certainty::Estimate,
            candidates,
            pv: Vec::new(),
            config,
        })
        .ok_or_else(|| ApiError::unprocessable("beam search found no move"))
}

fn seconds(milliseconds: Option<u64>) -> Option<f64> {
    milliseconds.map(|value| value.clamp(1, 300_000) as f64 / 1_000.0)
}

fn action_index(mv: &Move) -> u8 {
    mv.shape * 16 + mv.position
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    error: String,
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, message)
    }
    fn unprocessable(message: impl Into<String>) -> Self {
        Self::new(StatusCode::UNPROCESSABLE_ENTITY, message)
    }
    fn not_found(message: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_FOUND, message)
    }
    fn internal(message: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, message)
    }
    fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(ErrorBody {
                error: self.message,
            }),
        )
            .into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, http::Request};
    use http_body_util::BodyExt;
    use serde_json::{json, Value};
    use tower::ServiceExt;

    async fn json_response(response: Response) -> Value {
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn health_identifies_service_and_core_revision() {
        let response = app()
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = json_response(response).await;
        assert_eq!(body["service"], "quantik-api");
        assert_eq!(body["core_revision"], CORE_REVISION);
    }

    #[tokio::test]
    async fn minimax_returns_a_legal_portable_action() {
        let qfen = "AbC./..../..../....";
        let legal = generate_legal_moves(&State::from_qfen(qfen).unwrap().bb)
            .iter()
            .map(action_index)
            .collect::<Vec<_>>();
        let request = json!({
            "schema": REQUEST_SCHEMA,
            "qfen": qfen,
            "side_to_move": 1,
            "legal_action_indices": legal,
            "config": { "max_depth": 2, "seed": 7 }
        });
        let response = app()
            .oneshot(
                Request::post("/v1/move/minimax")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(request.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(json_response(response).await["action_index"], 51);
    }

    const OPENING_QFEN: &str = "AbC./..../..../....";

    async fn post_move(schema: &str) -> Response {
        let legal = generate_legal_moves(&State::from_qfen(OPENING_QFEN).unwrap().bb)
            .iter()
            .map(action_index)
            .collect::<Vec<_>>();
        let request = json!({
            "schema": schema,
            "qfen": OPENING_QFEN,
            "side_to_move": 1,
            "legal_action_indices": legal,
            "config": { "max_depth": 2, "seed": 7 }
        });
        app()
            .oneshot(
                Request::post("/v1/move/minimax")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(request.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn legacy_request_schema_is_still_accepted() {
        let response = post_move(REQUEST_SCHEMA_LEGACY).await;
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn unknown_request_schema_version_is_rejected() {
        let response = post_move("engine-request.v2").await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let error = json_response(response).await["error"]
            .as_str()
            .unwrap()
            .to_owned();
        assert!(error.contains(REQUEST_SCHEMA));
        assert!(!error.contains(REQUEST_SCHEMA_LEGACY));
    }

    #[tokio::test]
    async fn response_carries_the_bare_registered_schema_name() {
        // Even a legacy-spelled request is answered with the bare name.
        let response = post_move(REQUEST_SCHEMA_LEGACY).await;
        assert_eq!(
            json_response(response).await["schema"],
            "engine-response.v2"
        );
    }

    /// The registered v2 validator, or `None` when the sibling contracts
    /// checkout is absent and `QUANTIK_SKIP_CONTRACT_TESTS=1`.
    fn v2_validator() -> Option<jsonschema::Validator> {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../quantik-core-contracts/schemas/engine-response-v2.json"
        );
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(_) if std::env::var("QUANTIK_SKIP_CONTRACT_TESTS").as_deref() == Ok("1") => {
                eprintln!("SKIPPED: {path} not found and QUANTIK_SKIP_CONTRACT_TESTS=1");
                return None;
            }
            Err(error) => panic!(
                "{path} not readable ({error}); check out quantik-core-contracts as a sibling, \
                 or set QUANTIK_SKIP_CONTRACT_TESTS=1 to skip this test"
            ),
        };
        let schema: Value = serde_json::from_str(&text).unwrap();
        Some(jsonschema::validator_for(&schema).unwrap())
    }

    async fn move_body(engine: &str, qfen: &str, config: Value) -> Value {
        let state = State::from_qfen(qfen).unwrap();
        let legal_moves = generate_legal_moves(&state.bb);
        let legal = legal_moves.iter().map(action_index).collect::<Vec<_>>();
        let request = json!({
            "schema": REQUEST_SCHEMA,
            "qfen": qfen,
            "side_to_move": legal_moves[0].player,
            "legal_action_indices": legal,
            "config": config
        });
        let response = app()
            .oneshot(
                Request::post(format!("/v1/move/{engine}"))
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(request.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK, "{engine}");
        json_response(response).await
    }

    /// What the contract leaves to producers or to `validate_contracts.py`
    /// rather than to JSON Schema.
    fn assert_contract_extras(engine: &str, qfen: &str, body: &Value) {
        let legal = generate_legal_moves(&State::from_qfen(qfen).unwrap().bb)
            .iter()
            .map(|mv| u64::from(action_index(mv)))
            .collect::<Vec<_>>();
        let selected = body["action_index"].as_u64().unwrap();
        if let Some(pv) = body.get("pv") {
            assert_eq!(pv[0].as_u64().unwrap(), selected, "{engine}: pv[0]");
        }
        if let Some(candidates) = body.get("candidates") {
            let list = candidates.as_array().unwrap();
            assert!(list.len() <= MAX_CANDIDATES, "{engine}: too many");
            let indices = list
                .iter()
                .map(|c| c["action_index"].as_u64().unwrap())
                .collect::<Vec<_>>();
            assert_eq!(
                indices[0], selected,
                "{engine}: rank 0 is the selected move"
            );
            let mut unique = indices.clone();
            unique.sort_unstable();
            unique.dedup();
            assert_eq!(unique.len(), indices.len(), "{engine}: duplicates");
            assert!(indices.iter().all(|i| legal.contains(i)), "{engine}: legal");
        }
    }

    #[tokio::test]
    async fn response_validates_against_the_registered_schema() {
        let Some(validator) = v2_validator() else {
            return;
        };
        for engine in ["minimax", "mcts", "beam"] {
            let config = json!({ "max_depth": 2, "iterations": 50, "beam_width": 4, "seed": 7 });
            let body = move_body(engine, OPENING_QFEN, config).await;
            assert!(
                validator.is_valid(&body),
                "{engine} response violates engine-response-v2.json: {body}"
            );
            assert_contract_extras(engine, OPENING_QFEN, &body);
        }
    }

    #[tokio::test]
    async fn registered_v2_fixtures_validate_so_the_validator_is_not_vacuous() {
        let Some(validator) = v2_validator() else {
            return;
        };
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../quantik-core-contracts/fixtures/engine-response/engine-response-v2-synthetic.jsonl"
        );
        let text = std::fs::read_to_string(path).unwrap();
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let row: Value = serde_json::from_str(line).unwrap();
            assert!(validator.is_valid(&row), "fixture row rejected: {line}");
        }
        let mut missing = json!({
            "schema": "engine-response.v2", "action_index": 1, "engine_kind": "minimax",
            "engine_version": "x", "elapsed_ms": 1
        });
        assert!(!validator.is_valid(&missing), "certainty is required");
        missing["certainty"] = json!("estimate");
        assert!(validator.is_valid(&missing));
    }

    #[tokio::test]
    async fn every_engine_states_its_certainty_and_config() {
        for engine in ["minimax", "mcts", "beam"] {
            let config = json!({ "max_depth": 2, "iterations": 50, "beam_width": 4, "seed": 7 });
            let body = move_body(engine, OPENING_QFEN, config).await;
            assert!(
                matches!(body["certainty"].as_str(), Some("estimate" | "proof")),
                "{engine}: {body}"
            );
            assert!(body["engine_config"]
                .as_str()
                .is_some_and(|c| !c.is_empty()));
            assert_eq!(body["engine_version"], CORE_REVISION);
        }
    }

    #[tokio::test]
    async fn minimax_returns_its_principal_variation_and_value_candidates() {
        let body = move_body("minimax", QUIET_QFEN, json!({ "max_depth": 3, "seed": 7 })).await;
        let pv = body["pv"].as_array().expect("minimax carries a pv");
        assert!(!pv.is_empty());
        assert_eq!(pv[0], body["action_index"]);
        let candidates = body["candidates"].as_array().unwrap();
        assert!(candidates.iter().all(|c| c["unit"] == "value"));
        assert_eq!(candidates[0]["score"], body["value"]);
        assert_eq!(body["certainty"], "estimate", "depth 3 is heuristic");
        assert_eq!(body["engine_config"], "depth=3");
    }

    #[tokio::test]
    async fn mcts_returns_visit_candidates_and_no_pv() {
        let body = move_body(
            "mcts",
            OPENING_QFEN,
            json!({ "iterations": 200, "seed": 7 }),
        )
        .await;
        let candidates = body["candidates"].as_array().unwrap();
        assert!(candidates.iter().all(|c| c["unit"] == "visits"));
        assert!(
            candidates.iter().all(|c| c["score"].is_u64()),
            "integer counts"
        );
        assert_eq!(body["certainty"], "estimate");
        assert!(body.get("pv").is_none());
        assert_eq!(body["engine_config"], "iterations=200");
    }

    #[tokio::test]
    async fn beam_returns_a_pv_and_an_estimate() {
        let body = move_body("beam", OPENING_QFEN, json!({ "beam_width": 4, "seed": 7 })).await;
        assert_eq!(body["pv"][0], body["action_index"]);
        assert_eq!(body["certainty"], "estimate");
        assert_eq!(body["engine_config"], "beam_width=4");
    }

    /// Row 0 holds A, b, C and the side to move (player 1) completes the line
    /// with `d`: a mate in one, so minimax proves it at any depth.
    const MATE_IN_ONE_QFEN: &str = OPENING_QFEN;
    /// One piece down: no forced result within a shallow search.
    const QUIET_QFEN: &str = "A.../..../..../....";

    #[tokio::test]
    async fn a_mate_in_range_is_a_proof_and_drops_heuristic_candidates() {
        let config = json!({ "max_depth": 2, "seed": 7 });
        let body = move_body("minimax", MATE_IN_ONE_QFEN, config).await;
        assert_eq!(body["certainty"], "proof");
        assert_eq!(body["value"], 1.0);
        for c in body["candidates"].as_array().unwrap() {
            assert_eq!(
                c["score"].as_f64().unwrap().abs(),
                1.0,
                "only proven entries"
            );
        }
        if let Some(validator) = v2_validator() {
            assert!(validator.is_valid(&body), "{body}");
        }
        assert_contract_extras("minimax", MATE_IN_ONE_QFEN, &body);
    }

    #[tokio::test]
    async fn a_heuristic_search_is_never_a_proof() {
        let body = move_body("minimax", QUIET_QFEN, json!({ "max_depth": 2, "seed": 7 })).await;
        assert_eq!(body["certainty"], "estimate");
        assert!(body["value"].as_f64().unwrap().abs() < 1.0);
        if let Some(validator) = v2_validator() {
            assert!(validator.is_valid(&body), "{body}");
        }
        assert_contract_extras("minimax", QUIET_QFEN, &body);
    }

    #[test]
    fn candidates_are_legal_distinct_capped_and_lead_with_the_selected_move() {
        let mv = |shape, position| Move {
            player: 0,
            shape,
            position,
        };
        let ranked: Vec<(Move, Score, Unit)> = (0..12u8)
            .map(|i| (mv(0, i), Score::Count(100 - u64::from(i)), Unit::Visits))
            .chain([(mv(0, 3), Score::Count(1), Unit::Visits)])
            .collect();
        let legal: Vec<u8> = (0..12).filter(|i| *i != 1).collect();
        let list = sanitise_candidates(&ranked, &legal, 4).unwrap();
        assert_eq!(list.len(), MAX_CANDIDATES);
        assert_eq!(list[0].action_index, 4);
        assert!(list.iter().all(|c| c.action_index != 1), "illegal dropped");
        let mut seen: Vec<u8> = list.iter().map(|c| c.action_index).collect();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), list.len());
        assert!(sanitise_candidates(&ranked, &legal, 40).is_none());
    }

    #[tokio::test]
    async fn request_legality_must_match_core() {
        let request = json!({
            "schema": REQUEST_SCHEMA,
            "qfen": "..../..../..../....",
            "side_to_move": 0,
            "legal_action_indices": [0]
        });
        let response = app()
            .oneshot(
                Request::post("/v1/move/mcts")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(request.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }
}
