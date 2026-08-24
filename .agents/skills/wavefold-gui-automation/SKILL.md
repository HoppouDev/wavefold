---
name: wavefold-gui-automation
description: Drive or verify wavefold's iced desktop GUI programmatically, over its built-in localhost automation IPC — not OS-level screenshots or input-injection (xdotool/ydotool), which don't work reliably against a native GPU-rendered window and were tried and abandoned for exactly that reason. Use this whenever asked to test, verify, click through, automate, or interact with the wavefold app's UI — e.g. "does the encoder dropdown work", "walk through setting up an encode and check the state", "add a GUI test for the cutoff slider", "confirm the Start button is disabled until both files are picked". Also use when asked to add end-to-end or GUI-level tests to the wavefold project, since this IPC is the app's real, intended automation surface.
---

# wavefold GUI automation

wavefold's iced GUI can be driven and inspected over a localhost TCP JSON
protocol instead of real mouse/keyboard input. This exists because OS-level
automation (`xdotool`, `ydotool` + screenshots) was tried first and hit real,
unfixable problems on this app specifically: it's a native Wayland surface
invisible to X11 tools, KDE's focus-stealing prevention meant "active
window" screenshots often grabbed the wrong thing (including, once, the
user's private desktop — the reason this approach was abandoned), and
`ydotool`'s absolute positioning needed a touch-capable virtual device this
system's daemon couldn't create. None of that is a wavefold problem to fix;
driving the app's own message loop directly sidesteps all of it and works
identically on Linux/Windows/macOS.

## How it works

The GUI is an Elm-style app (`src/ui/`): every interaction is a `Message`
value fed into `update(&mut self, message) -> Task<Message>`, which is the
*only* thing that changes state. The automation server (`src/ui/automation.rs`,
behind the `automation` Cargo feature, off by default) exposes exactly that
same `Message` type over the wire — sending a message over the socket is
indistinguishable, from the app's perspective, from a real widget producing
it. There is no separate "automation API" to drift out of sync with the
real UI as it evolves.

## Setup

Build and launch with the feature enabled — it does nothing (no port opened)
in a normal build:

```bash
cargo build --release --features automation
./target/release/wavefold gui &
```

Wait ~1-2 seconds for the window and server to come up, then confirm it's
listening before sending anything:

```bash
python3 .agents/skills/wavefold-gui-automation/scripts/wavefold_automation.py snapshot
```

## Driving it

Use the bundled client (`scripts/wavefold_automation.py`) rather than
hand-rolling socket code — it handles the newline-framing and connection
errors:

```bash
# Read-only: current state, no side effect.
wavefold_automation.py snapshot

# Inject a real Message and get back the state *after* it was applied
# (synchronized server-side — never a race, never a guess).
wavefold_automation.py inject '{"Setup": {"CutoffChanged": 0.85}}'
```

If you need the raw protocol (e.g. from another language): newline-delimited
JSON on `127.0.0.1:47624`. Each line is either the bare string `"snapshot"`,
or `{"inject": <Message>}`. Every line sent gets exactly one JSON line back.

## Message reference

Top-level shape is `{"Setup": <setup message>}` or `{"Encoding": <encoding
message>}`, matching whichever screen is currently active (check `snapshot`
first if unsure — the response's `"screen"` field tells you). A unit
variant (no data) is just its bare name as a string, e.g. `"Start"`, so the
full injected message is `{"Setup": "Start"}`.

**Setup screen** (`src/ui/setup.rs`):

| Message | Shape | Notes |
|---|---|---|
| Set input path | `{"Setup": {"InputPicked": "/abs/path.mp4"}}` | Send this directly, not `PickInput` — that opens a real OS file dialog automation can't drive. |
| Set output path | `{"Setup": {"OutputPicked": "/abs/path.mp4"}}` | Same reasoning as above, not `PickOutput`. |
| Set cutoff | `{"Setup": {"CutoffChanged": 0.85}}` | `f32`, valid range `0.0..=2.0`. |
| Set encoder | `{"Setup": {"EncoderSelected": "H265Hardware"}}` | One of `H264`, `H264Hardware`, `H265`, `H265Hardware`, `Vp9`, `Vp9Hardware`, `Av1`, `Av1Hardware`. |
| Set compute backend | `{"Setup": {"BackendSelected": "Gpu"}}` | `"Gpu"` or `"Cpu"`. |
| Set DCT algorithm | `{"Setup": {"DctAlgorithmSelected": "Fft"}}` | `"Fft"` or `"Matmul"`; GPU-only, harmlessly ignored under `Cpu`. |
| Start the encode | `{"Setup": "Start"}` | No-ops if input or output isn't set yet (mirrors the real Start button being disabled) - check the response's `screen` to see whether it actually transitioned to `"encoding"`. |

**Encoding screen** (`src/ui/encoding.rs`):

| Message | Shape | Notes |
|---|---|---|
| Back to setup | `{"Encoding": "BackToSetup"}` | Only takes effect once `running` is `false` in the snapshot, same as the real button. |

`Pipeline`/`WorkerDone` messages arrive on their own from the real
background encode thread - don't inject them. Just poll `snapshot`
repeatedly to watch real progress/log/errors land, exactly as a human
watching the window would see them.

## Snapshot fields

**Setup**: `input` (nullable path string), `output` (nullable path string),
`cutoff` (f32), `encoder`, `backend`, `dct_algorithm` (all as their string
names above).

**Encoding**: `progress_current`, `progress_total` (u64; total is `0` until
known), `log` (array of strings, in order), `running` (bool).

## Example: full setup-to-encode sequence

```bash
S=.agents/skills/wavefold-gui-automation/scripts/wavefold_automation.py
python3 $S inject '{"Setup": {"InputPicked": "/tmp/in.mp4"}}'
python3 $S inject '{"Setup": {"OutputPicked": "/tmp/out.mp4"}}'
python3 $S inject '{"Setup": {"CutoffChanged": 0.4}}'
python3 $S inject '{"Setup": {"EncoderSelected": "Av1Hardware"}}'
python3 $S inject '{"Setup": "Start"}'   # -> screen should now be "encoding"

# Poll until done, watching the real background encode:
while true; do
  out=$(python3 $S snapshot)
  echo "$out"
  python3 -c "import json,sys; sys.exit(0 if not json.loads(sys.argv[1])['running'] else 1)" "$out" && break
  sleep 0.5
done
```

## Verifying visually (optional, rarely needed)

State assertions via `snapshot` are almost always sufficient and much more
reliable than pixels. If you genuinely need to see the rendered window
(e.g. debugging a layout issue), KDE Plasma/KWin sessions can capture just
that window safely via a KWin script that explicitly activates it by
`resourceClass` before calling `spectacle -b -a -n -o file.png` — do *not*
screenshot "whatever's active" without first pinning it that way, since a
background-launched window doesn't reliably get focus and an unscoped
capture can grab unrelated windows on the user's desktop. This is
genuinely last-resort territory; reach for `snapshot` first.
