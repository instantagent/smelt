#!/usr/bin/env python3
"""Per-trial exec->first-token timer for the cold-start receipt harness.

Measures, with a monotonic wall clock, the interval from just-before-process-spawn
to the first byte of the child's generated output on stdout. The child is
configured (by the caller) to emit exactly one generated token, and to send all
log/preamble noise to stderr so the first stdout byte is the first token.

For contenders that print an unavoidable stdout preamble before generation
(e.g. mlx_lm prints a "==========" banner), pass --start-after "<literal>":
the first-token clock is then taken at the first stdout byte that arrives AFTER
the FIRST occurrence of that literal is seen in the stream (this matches the
code below -- it triggers on the first match, not the last). t0 (spawn) is
unchanged. NOTE: when the child block-buffers its stdout to a pipe (mlx_lm
does), the preamble is NOT flushed early -- it lands in the SAME write as the
first token, so the marker only trims the preamble bytes within that first
flush; ttft still stamps exec->first-real-token because that flush happens at
first-token time.

Output (one key=value line to this process's stdout):
    ttft_ms=<float|nan> exit_ms=<float> rc=<int> stdout_bytes=<int> first=<repr>

Exit status (fail-closed): the line above ALWAYS prints (the raw retains the
evidence), but the timer then exits nonzero (3) if the child rc != 0, no first
byte was stamped (ttft=nan, e.g. marker never seen), or stdout_bytes == 0 — so
a failed trial cannot look like success to the harness.

The child's stderr is streamed to --err for later parsing (smelt Timing line,
TTS TTFA, startup trace). A short preview of stdout is written to --out to
verify a real token was produced (n=1 outputs are tiny).
"""
import argparse
import subprocess
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--err", required=True)
    ap.add_argument("--start-after", default=None)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("timer: empty command", file=sys.stderr)
        return 2

    marker = args.start_after.encode() if args.start_after else None
    errf = open(args.err, "wb")

    t0 = time.monotonic()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errf, bufsize=0)

    first = None
    seen = bytearray()
    marker_hit = marker is None
    preview = bytearray()
    total = 0
    assert p.stdout is not None
    while True:
        b = p.stdout.read(1)
        if not b:
            break
        total += 1
        if len(preview) < 48:
            preview += b
        if not marker_hit:
            seen += b
            if len(seen) > len(marker):
                del seen[:len(seen) - len(marker)]
            if bytes(seen) == marker:
                marker_hit = True
            continue
        if first is None:
            first = time.monotonic()

    p.wait()
    t_exit = time.monotonic()
    errf.close()

    with open(args.out, "wb") as of:
        of.write(bytes(preview))

    ttft_ms = (first - t0) * 1000.0 if first is not None else float("nan")
    exit_ms = (t_exit - t0) * 1000.0
    prev = bytes(preview).decode("utf-8", "replace").replace("\n", "\\n")
    print(f"ttft_ms={ttft_ms:.1f} exit_ms={exit_ms:.1f} rc={p.returncode} "
          f"stdout_bytes={total} first={prev!r}")
    # Fail-closed: the line above always prints (raw keeps the evidence), but a
    # failed trial must not look like success to the harness.
    if p.returncode != 0 or first is None or total == 0:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
