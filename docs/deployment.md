# Which container is the public deployment

Status: **decision paper, recommendations not yet accepted.** QW-020 criterion 4 asks for a
recorded answer; this file records the question, the options, and a recommendation for each
decision. Once the owner accepts or amends a recommendation, change the marker on that decision
from `RECOMMENDED` to `DECIDED` and keep the rejected alternatives.

Home: this file rather than `README.md`, because the README documents how to run the service and
this documents where it does and does not run in public. The README should link here (a one-line
follow-up, kept out of this change).

Two images exist or will exist:

- **`ghcr.io/mberlanda/quantik-api`** (this repository) — the Rust gateway. Classical engines only
  (`minimax`, `mcts`, `beam`), routes under `/v1/`.
- **`ghcr.io/mberlanda/quantik-models-play`** (`quantik-models-py`, QW-009) — the Python play
  service. Serves the trained networks, the classical engines, and the visualizer's static app from
  one port, with `--no-store`.

Evidence citations are `path:line`. Paths without a repository prefix are in `quantik-api-rust` at
`1764987` (origin/main); other repositories are cited at their working tree on 2026-09-20.

## Decisions needed

1. **Which image is the public deployment?** RECOMMENDED: the Python play container only.
2. **What is the Rust image then?** RECOMMENDED: a published distribution artifact for
   self-hosting the classical engines, explicitly not a public deployment.
3. **Which endpoint does the visualizer point at by default?** RECOMMENDED: same origin as the page,
   i.e. the play container; no default for a Remote seat.
4. **Where is a bug report filed?** RECOMMENDED: one owner per symptom, routed as below, with
   `quantik-models-py` owning anything seen on the public URL.
5. **What must change before the Rust image may ever be exposed publicly?** RECOMMENDED: name the
   preconditions now, do not meet them now.

## Decision 1: which image is the public deployment

What the evidence says:

- The play container serves the page and the API on one port, which is what a shareable URL needs.
  It runs `--runtime onnx --no-store` (`quantik-models-py/docker/Dockerfile:68-71`, `EXPOSE 8000` at
  `:59`) and serves the visualizer as package data
  (`quantik-models-py/docs/play-service.md:232-235`).
- The Rust image serves only JSON. Its router has `/health`, `/v1/engines`, `/v1/move/{engine}`
  (`src/lib.rs:32-34`) and no static route, so a visitor has nothing to open in a browser. Making it
  public would need a second host for the page.
- The Rust image cannot serve a trained network. `quantik-models-py/docs/play-service.md:407-409`
  says so, and `docs/model-serving.md:6` records that the Rust-side inference decision is still
  unmade. `QW-020/decisions.md:24` says the two are "not interchangeable — QW-009 serves the
  networks today and this does not".
- The stated audience is people who want to play a skill level, with network internals as an
  advanced mode (memory: `quantik-public-deployment-goal`). Only the play container has both the
  networks and the classical engines behind one opponent list.
- ADR 0012 (`quantik-workspace/docs/adr/0012-storeless-first-public-deployment.md:7-10`) records
  the public posture as `--no-store`, a flag of the Python service, and its consequence at `:19-21`
  (the client learns from `GET /api` that no store exists) is Python-API behaviour with no Rust
  counterpart. The Rust gateway has no store at all, so it is storeless trivially, not by decision.

Options:

- **A. Python play container only. RECOMMENDED.** One URL, one owner, the whole product visible.
  Cost: the image is ~498 MB (`quantik-models-py/docs/play-service.md:326-328`) against 10.7 MB for
  the Rust image (`QW-020/status.md`, W1 entry), and it carries CC-BY-NC-4.0 weights, so the public
  deployment inherits a non-commercial constraint (`play-service.md:366-368`). ADR 0012's direction
  already accepts a weights-carrying image.
- **B. Both are public: Rust for classical engines at one URL, Python for networks at another.**
  Gives third parties a cheap, permissively licensed endpoint. Cost: two public URLs, two CORS
  postures, two surfaces to keep up, and exactly the "two public deployments with no stated owner"
  outcome this criterion exists to prevent. The Python service's roster already includes classical
  engines (`quantik-models-py/docs/play-service.md:16`; it names `minimax-d2` at `:172`), so a
  visitor gains nothing from the second URL.
- **C. Rust image only, with the visualizer hosted separately.** Smallest, MIT only. Cost: no
  trained opponents, which is the point of the project, and a second static host.

Rejected: B on ownership, C on capability. Neither is a technical failure, which is why the
rejection is written down: if Rust gains model serving, reopen this decision rather than assume it.

Relationship to ADR 0012: this sits inside it. It adds no store, does not change the storeless
default, and creates no second public posture. It does not amend the ADR; it names which artifact
the ADR's "public deployment" refers to.

## Decision 2: what the Rust image is

Options:

- **A. A distribution artifact for self-hosters and for the smoke test. RECOMMENDED.** Keep
  publishing to GHCR (`.github/workflows/publish-image.yml`: a `vX.Y.Z` tag pushes
  `quantik-api:X.Y.Z`, `latest` deliberately not pushed, `:31-32`). Its purpose: a small,
  dependency-free classical-engine gateway, a container for the visualizer's Remote mode
  (`README.md:30-34`), and the artifact QW-020 W3 tests. It carries the core revision as provenance
  (`src/lib.rs:21`, returned by `/health` at `:52-57`).
- **B. Do not publish it; keep the Dockerfile for local use.** Rejected: W1 to W3 exist, the image is
  10.7 MB, and shipping the base image before QW-017 is already recorded
  (`QW-020/decisions.md:11-14`).
- **C. Publish it and call it a public deployment "for engines".** Rejected for decision 1's option
  B reason: the phrase invites people to point users at it.

Wording consequence: the image description and README say "self-hosted classical-engine gateway",
never "public deployment". Per `QW-020/decisions.md:16-19`, nothing here claims the image defines
the engine interface; `engine-request` and `engine-response` do.

## Decision 3: the visualizer's default endpoint

The visualizer already has no fixed default. The play-service base is "same origin as this page"
unless overridden (`quantik-qfen-visualizer/index.html:207-212`, `src/app.js:185-191`), and a
Remote seat's endpoint is an empty field with only a placeholder (`index.html:197`,
`src/app.js:266-269`). Those placeholders (`localhost:8000/move`, `localhost:8001/move`) match
neither service's real routes.

Options:

- **A. Default is same origin, meaning the play container when it serves the page; Remote seats stay
  opt-in and empty. RECOMMENDED.** This is what the code does today, so recording it costs nothing.
  The public URL needs no configuration and nobody is sent to the Rust image by default. The Rust
  image is reached only by pasting `http://<host>:<port>/v1/move/mcts` into a Remote seat
  (`README.md:30-34`).
- **B. Hard-code a public URL for the Rust gateway in the visualizer.** Rejected: a hidden dependency
  from a dependency-free app onto a hosted service, and it contradicts decision 1.
- **C. Fix the Remote placeholders to the real routes.** Not decided here; the visualizer is outside
  this work item's `allowed_paths`. Noted as a follow-up in that repository.

Port note, not a decision: the Rust binary defaults to 8000 (`src/main.rs:13`, `README.md:19`), the
image overrides it to 8080 (`Dockerfile:27-28`), and the play container uses 8000
(`quantik-models-py/docker/Dockerfile:59`). Anyone running both on one host remaps one.

## Decision 4: where a bug report goes

The rule: the artifact that produced the behaviour owns the bug, and the public URL is always
`quantik-models-py`'s. This is the answer criterion 4 requires when both images ship.

- Seen on the public URL or in the play container (wrong opponent, missing model, the page, a 503
  from `POST /api/games` in storeless mode): `quantik-models-py`.
- Seen running `quantik-api` yourself, or an image from `ghcr.io/mberlanda/quantik-api`:
  `quantik-api-rust`.
- The two images disagree on the same position: `quantik-core-contracts` first, since the contract
  is authoritative (`QW-020/decisions.md:16-19`); triage reassigns.
- Legality, QFEN or search behaviour in the engines themselves: `quantik-core-rust` or
  `quantik-core-py`, reassigned by whoever triages.

Options:

- **A. The routing above. RECOMMENDED.** One paragraph per README; there is always a single default
  owner.
- **B. Everything to `quantik-workspace`.** One inbox, but it is the control plane rather than a
  product repository and would become a pass-through.
- **C. Always `quantik-api-rust` because it is the gateway.** Rejected: it does not serve the public
  page and cannot debug a model.

## Decision 5: what would have to be true to expose the Rust image publicly

Not proposed for now. Writing it down keeps decision 1 from being reopened by accident:

- CORS allows any origin (`src/lib.rs:36-39`), documented as development-only with an instruction to
  restrict at the proxy or router before public exposure (`README.md:41-42`). Restrict first.
- The repository has no CI, `linux/amd64` was never built or tested, and the publish workflow has
  never run (`QW-020/status.md`, W1 and W2 entries).
- `CORE_REV` must stay in sync with `CORE_REVISION` (`Dockerfile:13`, `src/lib.rs:21`).
- A named owner, and a reason a visitor would use it instead of the play container.

Options: **A.** record these preconditions and change nothing (RECOMMENDED); **B.** change the CORS
default now (rejected: outside `allowed_paths` and a code change in a docs PR); **C.** say nothing
(rejected: the README warning would be the only guard).

## Documents reconciled against

- `quantik-workspace/docs/adr/0012-storeless-first-public-deployment.md`
- `quantik-workspace/tasks/active/QW-020-rust-api-container-distribution/decisions.md` (`QW-020/decisions.md`
  above) and `status.md`
- `quantik-models-py/docs/play-service.md` (Docker and GHCR sections) and `docker/Dockerfile`
- `quantik-qfen-visualizer/index.html`, `src/app.js`, `README.md`
- `docs/model-serving.md` and `README.md` in this repository

Not verified: that either GHCR image is currently published, or that the play container's tags
match `quantik-models-py/docs/play-service.md:359-364`. Neither registry was queried.
