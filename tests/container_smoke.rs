//! Container smoke test (QW-020/W3): the image runs the engines it claims.
//!
//! Starts the image, reads the engine kinds it advertises from `/v1/engines`, posts one
//! request per kind, and asserts each returned action is in the request's own
//! `legal_action_indices`. A 200 alone proves nothing; an index outside that list is the defect.
//!
//! Build the image first: `docker build -t quantik-api:dev .` (override the tag with
//! `QUANTIK_API_IMAGE`). Without a reachable Docker daemon the test skips with a message.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::process::Command;
use std::time::{Duration, Instant};

use quantik_core::{moves::generate_legal_moves, state::State};
use serde_json::{json, Value};

const DEFAULT_IMAGE: &str = "quantik-api:dev";
const OPENING_QFEN: &str = "AbC./..../..../....";
const SIDE_TO_MOVE: u8 = 1;

fn image() -> String {
    std::env::var("QUANTIK_API_IMAGE").unwrap_or_else(|_| DEFAULT_IMAGE.to_owned())
}

fn docker_available() -> bool {
    Command::new("docker")
        .args(["info", "--format", "{{.ServerVersion}}"])
        .output()
        .map(|out| out.status.success())
        .unwrap_or(false)
}

/// A running container, removed on drop even if an assertion panics.
struct Container {
    id: String,
    port: u16,
}

impl Container {
    fn start(image: &str) -> Self {
        let out = Command::new("docker")
            .args(["run", "-d", "--rm", "-p", "127.0.0.1::8080", image])
            .output()
            .expect("failed to run docker");
        assert!(
            out.status.success(),
            "docker run {image} failed (did you `docker build -t {image} .`?): {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let id = String::from_utf8_lossy(&out.stdout).trim().to_owned();
        let mut container = Container { id, port: 0 };
        let mapping = Command::new("docker")
            .args(["port", &container.id, "8080/tcp"])
            .output()
            .expect("failed to run docker port");
        let text = String::from_utf8_lossy(&mapping.stdout);
        container.port = text
            .lines()
            .next()
            .and_then(|line| line.rsplit(':').next())
            .and_then(|port| port.trim().parse().ok())
            .unwrap_or_else(|| panic!("could not parse published port from {text:?}"));
        container
    }

    fn wait_healthy(&self) {
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            if let Ok((200, _)) = self.request("GET", "/health", None) {
                return;
            }
            assert!(Instant::now() < deadline, "container never became healthy");
            std::thread::sleep(Duration::from_millis(200));
        }
    }

    /// Minimal HTTP/1.1 client: returns (status, parsed JSON body).
    fn request(
        &self,
        method: &str,
        path: &str,
        body: Option<&Value>,
    ) -> Result<(u16, Value), String> {
        let mut stream = TcpStream::connect(("127.0.0.1", self.port)).map_err(|e| e.to_string())?;
        stream
            .set_read_timeout(Some(Duration::from_secs(60)))
            .map_err(|e| e.to_string())?;
        let payload = body.map(Value::to_string).unwrap_or_default();
        let request = format!(
            "{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\
             Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{payload}",
            payload.len()
        );
        stream
            .write_all(request.as_bytes())
            .map_err(|e| e.to_string())?;
        let mut raw = Vec::new();
        stream.read_to_end(&mut raw).map_err(|e| e.to_string())?;
        let text = String::from_utf8_lossy(&raw);
        let (head, rest) = text.split_once("\r\n\r\n").ok_or("malformed response")?;
        let status = head
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .ok_or("malformed status line")?;
        let rest = if head
            .to_ascii_lowercase()
            .contains("transfer-encoding: chunked")
        {
            dechunk(rest)
        } else {
            rest.to_owned()
        };
        let json = serde_json::from_str(&rest).map_err(|e| format!("{e}: {rest:?}"))?;
        Ok((status, json))
    }
}

impl Drop for Container {
    fn drop(&mut self) {
        let _ = Command::new("docker").args(["rm", "-f", &self.id]).output();
    }
}

fn dechunk(body: &str) -> String {
    let mut out = String::new();
    let mut rest = body;
    while let Some((size, tail)) = rest.split_once("\r\n") {
        let size = usize::from_str_radix(size.trim(), 16).unwrap_or(0);
        if size == 0 || tail.len() < size {
            break;
        }
        out.push_str(&tail[..size]);
        rest = tail[size..].trim_start_matches("\r\n");
    }
    out
}

/// The defect this test exists to catch: an action outside the request's declared legal set.
fn assert_legal(engine: &str, action: u64, legal: &[u8]) {
    assert!(
        legal.iter().any(|l| u64::from(*l) == action),
        "{engine} returned action {action}, which is not in legal_action_indices {legal:?}"
    );
}

fn legal_actions(qfen: &str) -> Vec<u8> {
    let state = State::from_qfen(qfen).expect("valid QFEN");
    let mut actions: Vec<u8> = generate_legal_moves(&state.bb)
        .iter()
        .map(|mv| mv.shape * 16 + mv.position)
        .collect();
    actions.sort_unstable();
    actions.dedup();
    actions
}

#[test]
#[should_panic(expected = "not in legal_action_indices")]
fn legality_assertion_rejects_an_illegal_index() {
    let legal = legal_actions(OPENING_QFEN);
    let illegal = (0..64u64)
        .find(|a| !legal.contains(&(*a as u8)))
        .expect("some index is illegal");
    assert_legal("probe", illegal, &legal);
}

#[test]
fn every_advertised_engine_returns_a_legal_move() {
    if !docker_available() {
        eprintln!("SKIPPED: docker is unavailable; container smoke test not run");
        return;
    }
    let image = image();
    let container = Container::start(&image);
    container.wait_healthy();

    let (status, engines) = container
        .request("GET", "/v1/engines", None)
        .expect("GET /v1/engines");
    assert_eq!(status, 200, "/v1/engines: {engines}");
    let kinds: Vec<String> = engines
        .as_array()
        .expect("/v1/engines returns an array")
        .iter()
        .map(|e| e["kind"].as_str().expect("engine has a kind").to_owned())
        .collect();
    assert!(!kinds.is_empty(), "/v1/engines advertised no engines");
    eprintln!("image {image}: engine kinds {kinds:?}");

    let legal = legal_actions(OPENING_QFEN);
    for kind in &kinds {
        let request = json!({
            "schema": "engine-request.v1",
            "qfen": OPENING_QFEN,
            "side_to_move": SIDE_TO_MOVE,
            "legal_action_indices": legal,
            "config": { "max_depth": 2, "iterations": 50, "beam_width": 4, "seed": 7 }
        });
        let (status, body) = container
            .request("POST", &format!("/v1/move/{kind}"), Some(&request))
            .unwrap_or_else(|e| panic!("POST /v1/move/{kind}: {e}"));
        assert_eq!(status, 200, "{kind}: {body}");
        assert_eq!(body["engine_kind"], kind.as_str(), "{kind}: {body}");
        let action = body["action_index"]
            .as_u64()
            .unwrap_or_else(|| panic!("{kind}: no action_index in {body}"));
        eprintln!("  {kind}: action_index {action} (legal: {legal:?})");
        assert_legal(kind, action, &legal);
    }
}
