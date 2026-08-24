#!/usr/bin/env python3
"""Thin client for wavefold's GUI automation IPC (see ../SKILL.md).

Talks newline-delimited JSON to 127.0.0.1:47624, which only exists once a
wavefold binary built with `--features automation` is running as `gui`.

Usage:
    wavefold_automation.py snapshot
    wavefold_automation.py inject '{"Setup": {"CutoffChanged": 0.85}}'
    wavefold_automation.py inject '{"Setup": "Start"}'
    echo '{"Setup": {"EncoderSelected": "H265Hardware"}}' | wavefold_automation.py inject -

Each subcommand prints the resulting state snapshot as one JSON line on
stdout and exits 0. Connection or protocol errors go to stderr with a
non-zero exit so this composes fine in a shell pipeline or a test script.
"""
import json
import socket
import sys

HOST, PORT = "127.0.0.1", 47624


def send(request_obj) -> dict:
    with socket.create_connection((HOST, PORT), timeout=5) as sock:
        sock.sendall((json.dumps(request_obj) + "\n").encode())
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = sock.recv(4096)
            if not chunk:
                raise ConnectionError("server closed the connection without a response")
            buf += chunk
        return json.loads(buf.decode())


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in ("snapshot", "inject"):
        print(__doc__, file=sys.stderr)
        return 2

    if sys.argv[1] == "snapshot":
        request = "snapshot"
    else:
        if len(sys.argv) < 3:
            print("inject needs a JSON message argument (or '-' to read one from stdin)", file=sys.stderr)
            return 2
        raw = sys.stdin.read() if sys.argv[2] == "-" else sys.argv[2]
        try:
            message = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"argument is not valid JSON: {e}", file=sys.stderr)
            return 2
        request = {"inject": message}

    try:
        result = send(request)
    except (OSError, ConnectionError) as e:
        print(f"could not reach wavefold automation server at {HOST}:{PORT}: {e}", file=sys.stderr)
        print("is wavefold running as `gui`, built with --features automation?", file=sys.stderr)
        return 1

    print(json.dumps(result))
    if "error" in result:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
