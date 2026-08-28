# Serving the trained model from the Rust API

Design note comparing two runtimes for policy/value inference inside
`quantik-api-rust`, so the visualizer can play against the network the way it
already plays against minimax, MCTS and beam.

Status: **decision not yet made.** This document exists to make it.

---

## Where things stand today

Three of the four pieces already exist, which is worth saying plainly before
comparing runtimes.

- **`quantik-qfen-visualizer` needs no work.** It already has a *Remote* player
  mode that POSTs `quantik.engine-request.v1` to an arbitrary endpoint. Any new
  engine route is reachable the moment it exists.
- **`quantik-api-rust` already is the gateway.** `POST /v1/move/{engine}` serves
  `minimax`, `mcts` and `beam`, recomputing legality with `quantik-core` before
  running anything, and returning `action_index = shape * 16 + position`.
- **`quantik-core-rust` has the classical engines** — `minimax.rs`, `mcts.rs`,
  `beam_search.rs` — and no neural inference of any kind.
- **`quantik-models-py` has the trained networks**, the architecture, a batched
  AlphaZero MCTS, and the two model-backed agents the articles measured.

The missing piece is inference in Rust. Everything below is about how to get it.

### A prerequisite that blocks all options

`quantik-models-py/.gitignore` contains `runs/`. Twelve checkpoints exist on the
training machine and **none of them is committed, released, or otherwise
retrievable**. Before the API can serve a model, the checkpoint has to be
something a deployment can fetch:

| checkpoint | architecture | parameters | weights |
|---|---|---|---|
| `sup-sampled-c128b6-best` | `resnet-c128-b6` | 1,786,823 | 6.8 MB |
| `sweep-c128b6-best` | `resnet-c128-b6` | 1,786,823 | 6.8 MB |
| `az-small-v1-best`, `sup-deep-smoke{,2}-best`, `sweep-c64b4-best` | `resnet-c64-b4` | 304,711 | 1.2 MB |
| `smoke-best` | `resnet-c16-b2` | 13,991 | 60 KB |

`sup-sampled-c128b6-best` is the 1.8M-parameter network from the Part VI and VII
drafts. `model-checkpoint.v1` already exists as a contract schema, so the
natural distribution is a GitHub Release on `quantik-models-py` carrying
`weights.safetensors` alongside its `manifest.json`. The 60 KB smoke checkpoint
is small enough to commit outright and makes a good CI fixture.

Two related snags: every manifest currently reads `contract_version: 1.1.0` and
should be regenerated at 1.2.0, and **`quantik-api-rust` has no git remote** —
it is two local commits that exist on one machine.

---

## What is the same in both options

Roughly eighty percent of this work does not depend on which runtime wins, which
is the main reason the decision is lower-stakes than it looks.

**The input encoding.** A `(9, 4, 4)` float32 tensor. Planes 0–7 are the
bitboard planes — two players by four shapes — set to 1.0 where a bit is set,
with `row = position / 4` and `col = position % 4`. Plane 8 is filled with the
side to move.

> **Correction.** An earlier version of this document said the planes must
> match `quantik_core.ml_data.qfen_to_tensor`. They must not. There are two
> distinct nine-plane layouts in this project, both legitimately
> `tensor-board.v1`, and the checkpoints were trained on the other one:
>
> - **colour-ordered** — planes 0–3 are player 0's shapes A–D, planes 4–7 are
>   player 1's. This is `qfen_to_tensor`, and also
>   `fastboard.to_core_tensor`. **Nothing in the training stack uses it.**
> - **mover-relative** — planes 0–3 are the *side to move*'s shapes A–D,
>   planes 4–7 the opponent's. This is `fastboard.encode_tensors`, and it is
>   what `train/supervised.py` and `eval/evaluator.py` both feed the network.
>
> Every published checkpoint is trained mover-relative. A runtime that
> encodes colour-ordered swaps the two players on every position where
> player 1 is to move — half of them — and the network answers a question
> about the wrong side. It will not error; it will just be wrong, and only
> on half the board states, which is the worst way for it to be wrong.

So: read the eight bitboards, and if the side to move is player 1, emit their
four shape planes first. Plane 8 carries the side-to-move flag either way.
`quantik-core-rust` already exposes the bitboard planes, so this is roughly
twenty lines of Rust — the permutation is the only subtlety.

The mover-relative layout is not an accident: it is what makes a *single*
value head with one sign convention learnable. A colour-ordered network would
have to learn "return +1 when player 0 wins, unless it is player 1 to move" as
a function of plane 8, rather than always answering "good for me?".

**Masking lives outside the model.** `model-checkpoint.v1` states it as a rule:
*"Runtimes must apply legal action masks outside the model."* Fill illegal
logits with the float32 minimum, then softmax. This is the same masking the
training loss applies, via the single shared `masked_log_softmax`.

**Move selection.** `net-policy` takes the argmax of the masked priors —
temperature 0, one forward pass, no search. `net-mcts` runs PUCT with the
network supplying priors and values, then takes the argmax of the visit counts.

**The MCTS port.** `net-mcts` needs `BatchedMCTS` — 330 lines of NumPy in
`quantik-models-py/src/quantik_models/selfplay/mcts.py` — reimplemented against
`quantik-core`'s board representation. **Neither runtime option avoids this.**
It is the single largest piece of work and it is identical either way.

---

## Option A — candle, Rust-native

`candle-core` and `candle-nn` 0.11.0, pure Rust, reading `weights.safetensors`
directly through a `VarBuilder`. No export step, no native library, no second
artifact: the file the trainer already writes is the file the server loads.

The architecture is small and entirely conventional, so the port is mechanical.
The checkpoint holds 96 tensors and their names map one-to-one onto the module
tree, which is what makes this predictable rather than fiddly:

```
stem.0.weight              conv 9 -> C, 3x3, bias=False
stem.1.{weight,bias,running_mean,running_var}      batch norm
trunk.{i}.conv1.weight     for i in 0..blocks
trunk.{i}.bn1.{...}
trunk.{i}.conv2.weight
trunk.{i}.bn2.{...}
policy_head.0.weight       conv C -> 2, 1x1        (0 conv, 1 bn, 2 relu,
policy_head.1.{...}                                 3 flatten, 4 linear)
policy_head.4.{weight,bias}  linear 32 -> 64
value_head.0.weight        conv C -> 1, 1x1        (0 conv, 1 bn, 2 relu,
value_head.1.{...}                                  3 flatten, 4 linear,
value_head.4.{weight,bias}   linear 16 -> 64        5 relu, 6 linear, 7 tanh)
value_head.6.{weight,bias}   linear 64 -> 1
```

Everything is F32 except the `num_batches_tracked` scalars, which are I64 and
are not parameters — inference ignores them. Batch norm at inference is an
affine transform over the stored running statistics, which `candle-nn` provides
directly.

**In favour.** One self-contained binary. No export step to forget and no second
artifact to drift out of sync with the safetensors. The manifest's
`weights_hash` verifies precisely the bytes being served. The architecture has
been stable across every checkpoint produced so far.

**Against.** The architecture is expressed twice, in Python and in Rust, and the
two can silently diverge — a wrong batch-norm epsilon or a transposed linear
produces plausible-looking moves rather than an error. This is a real risk and
it has a specific mitigation: a committed parity test that runs a set of fixed
positions through both and asserts the logits agree to a tolerance. That test is
not optional. If the architecture ever changes, Rust must change with it.

Roughly 150–200 lines for the network, plus the parity fixture.

---

## Option B — ONNX

Export each checkpoint once from PyTorch, then run the graph in Rust. The
contract already anticipates this: `weights_format` accepts `onnx`, and
`quantik-core`'s own `is_supported_weights_format` accepts
`"safetensors" | "onnx" | "npz" | "custom-binary"`. Adding
`torch.onnx.export` beside the existing `export_checkpoint` is small.

Two Rust runtimes, and the difference between them matters more than the choice
of ONNX itself:

| | `tract-onnx` 0.23.5 | `ort` 2.0.0-rc.13 |
|---|---|---|
| implementation | pure Rust, self-contained | wraps native ONNX Runtime 1.28 |
| deployment | nothing extra to ship | a native library must be present |
| release status | stable | **still a release candidate** |
| speed | slower, irrelevant for a 1.8M-parameter net on 4×4 | fastest |

For a network this small, `tract` is the sensible half of this option: pure Rust
keeps the single-binary property that makes candle attractive, without expressing
the architecture twice.

**In favour.** The architecture is defined once, in Python. Changing it — the
`target` preset is `resnet-c256-b13` at about 15.4M parameters — requires no Rust
change at all. The exported graph is the contract, so there is nothing to keep
in sync by hand.

**Against.** A second artifact per checkpoint, which must be exported, hashed and
published alongside the safetensors, and which can go stale. `weights_hash` then
covers the safetensors while the server actually runs the ONNX file, so the
manifest no longer describes what is being served unless the export pipeline is
disciplined about it. `ort` specifically would add a native dependency and a
pre-release crate to a service you want to deploy.

---

## What is reusable from quantik-models-py

This is where the two options genuinely differ, and it is narrower than expected.

| piece | lines | candle | ONNX |
|---|---|---|---|
| `model/policy_value_net.py` — the architecture | 97 | reimplemented in Rust | exported, not reimplemented |
| `selfplay/mcts.py` — `BatchedMCTS` | 330 | **ported to Rust** | **ported to Rust** |
| `selfplay/evaluator.py` — masking and batching | 78 | ported (thin) | ported (thin) |
| `arena/agents.py` — `PolicyAgent`, `NetMCTSAgent` | 207 | selection policy only | selection policy only |
| `qfen_to_tensor` in `quantik-core-py` | ~20 | ported | ported |

Only the first row changes. **ONNX saves the architecture port and nothing
else** — 97 lines of the roughly 500 that need porting. If the reason to prefer
ONNX is "reuse the Python work", that reason is weaker than it appears: the
expensive and error-prone part is the MCTS, and it is unavoidable in Rust either
way.

The one option that would reuse all of it is a Python sidecar serving
`PolicyAgent` and `NetMCTSAgent` over the same `engine-request.v1` protocol. The
visualizer would not know the difference. That was set aside because it leaves
the Rust API without a model engine, but it remains the cheapest path to
*playing* the model, and it is worth keeping as a stopgap while the Rust work
lands — it also doubles as the reference implementation the parity test compares
against.

---

## Routing: targeting a specific model

The requirement is to select a model from the route, not from a config file.
Current routing stays untouched; the model-backed engines take an optional model
segment:

```
GET  /v1/engines                     minimax, mcts, beam, net-policy, net-mcts
GET  /v1/models                      loaded checkpoints, from their manifests
POST /v1/move/{engine}               classical engines, or the default model
POST /v1/move/{engine}/{model_id}    a specific checkpoint
```

So `POST /v1/move/net-mcts/sup-sampled-c128b6-best` plays the 1.8M network with
search, and `POST /v1/move/net-policy/smoke-best` plays the 60 KB one with a
single forward pass. `model_id` is the field of the same name in
`model-checkpoint.v1`, so routes are addressed by the contract's own identifier
rather than by a filename or a path.

`GET /v1/models` returns what the manifest already carries — `model_id`,
`architecture`, `parameter_count`, `contract_version`, `weights_hash` — which is
enough for the visualizer to populate a dropdown without hardcoding anything.

**Registry.** A directory scanned at boot, `QUANTIK_MODEL_DIR`, one subdirectory
per checkpoint holding `manifest.json` and its weights. At load, assert
`schema == "model-checkpoint.v1"`, verify `weights_hash` against the file, and
refuse to serve a checkpoint that fails either — a mismatched hash is exactly the
failure mode that otherwise shows up as a subtly bad player. Serving no models is
a valid state: the classical routes keep working and `/v1/models` returns an
empty list.

**Response.** `MoveResponse` already carries `engine_kind` and `engine_version`.
For model engines, `engine_version` should be the `model_id` rather than the core
revision, so an exported game records which network produced it.

---

## Recommendation

**Take option A, candle, and write the parity test first.**

The architecture has been identical across all twelve checkpoints, it is small
and conventional, and candle reads the artifact the trainer already produces. The
single-binary property is worth real money for a service, and ONNX's advantage —
not expressing the architecture twice — buys 97 lines out of roughly 500 while
introducing a second artifact that the manifest's hash does not cover.

The decision is also cheaply reversible, because everything in *What is the same
in both options* is runtime-independent. If the `target` preset lands, or the
architecture starts varying per experiment, swapping candle for `tract` behind
the evaluator trait is a contained change.

Suggested order of work:

1. Publish `sup-sampled-c128b6-best` and `smoke-best` as a `quantik-models-py`
   release, regenerated at contract 1.2.0. Nothing can be served until this
   exists.
2. Give `quantik-api-rust` a git remote.
3. Encoder plus network in candle, with the parity test against PyTorch on fixed
   positions.
4. `net-policy`, the registry, and the `/v1/models` and `/v1/move/{engine}/{model_id}`
   routes. This is a playable slice.
5. Port `BatchedMCTS` and add `net-mcts`.

Steps 1 through 4 are the ones that put a network in front of the visualizer.
Step 5 is the larger half of the work and the one that produces the player the
articles actually measured.
