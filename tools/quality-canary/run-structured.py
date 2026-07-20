#!/usr/bin/env python3
"""Structured-task quality canary runner (stdlib only, no venv).

Drives an already-running `smelt serve --transport http` endpoint over
/v1/chat/completions with one fixed neutral prompt per case, greedy
(temperature=0) decoding, and scores JSON-extraction / tool-call cases against
ground truth.

Terminator discipline: every response's finish_reason is recorded. A canary is
only trustworthy if generation stops on the package's turn-end terminator; runs
that hit max_tokens ("length") or leak a raw <|im_start|>/<|im_end|> marker are
counted and surfaced loudly -- an incomplete terminator set silently runs past
turn end and corrupts the score.

Usage: run-structured.py --base-url URL --model ID --corpus FILE --out FILE
"""
import argparse
import json
import re
import sys
import urllib.request

# Tool catalog for tool_call cases: name -> ordered arg keys. Presented to the
# model as the available functions (standard function-calling setup, not tuning).
TOOL_CATALOG = {
    "get_weather": ["location"],
    "set_timer": ["minutes"],
    "play_song": ["title", "artist"],
    "add_to_list": ["item"],
    "translate": ["text", "target_language"],
    "convert_currency": ["amount", "from", "to"],
    "set_reminder": ["task", "time"],
    "control_lights": ["room", "state"],
    "web_search": ["query"],
    "book_table": ["party_size", "restaurant"],
}


def build_messages(case):
    if case["category"] == "tool_call":
        catalog = "\n".join(
            "- %s(%s)" % (n, ", ".join(a)) for n, a in TOOL_CATALOG.items())
        system = (
            "You are a function-calling engine. Given a user request, choose the "
            "single best function and its arguments from this catalog:\n" + catalog +
            "\nReply with ONLY a JSON object of the form "
            '{"name": <function>, "arguments": {<arg>: <value>}}. '
            "Copy argument values from the request. No commentary, no code fences.")
        user = case["text"]
    else:
        keys = ", ".join(case["fields"])
        system = (
            "You are a precise information-extraction engine. Read the text and "
            "reply with ONLY a single JSON object containing exactly these keys: " +
            keys + ". Copy the values from the text verbatim; use a number (not a "
            "string) for numeric fields. No commentary, no code fences.")
        user = case["text"]
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


def chat(base_url, model, messages, max_tokens):
    body = json.dumps({
        "model": model, "messages": messages,
        "temperature": 0, "top_p": 1, "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(base_url + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)


FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


def extract_json(text):
    """Parse the ENTIRE response as a single JSON object, or return None.

    Optionally strips one enclosing markdown code fence, then requires the whole
    remaining string to parse via json.loads. Trailing garbage after a balanced
    object (e.g. `{...})`) fails -- there is no "find first balanced object and
    ignore the rest". A non-object top-level value (array, number, ...) also
    fails: the task demands a single JSON object.
    """
    s = FENCE.sub("", text.strip())
    try:
        obj = json.loads(s)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def norm_str(v):
    return re.sub(r"\s+", " ", str(v)).strip().casefold()


def val_eq(exp, got):
    if isinstance(exp, bool):
        return isinstance(got, bool) and exp == got
    if isinstance(exp, (int, float)) and not isinstance(exp, bool):
        # Require a JSON numeric TYPE. A numeric string like "42" fails even if
        # float()-equal; the 1e-6 tolerance applies only after the type check.
        if not isinstance(got, (int, float)) or isinstance(got, bool):
            return False
        return abs(float(exp) - float(got)) < 1e-6
    if isinstance(exp, dict):
        if not isinstance(got, dict):
            return False
        return all(k in got and val_eq(exp[k], got[k]) for k in exp)
    return norm_str(exp) == norm_str(got)


def keys_exact(target, got):
    """True iff got has EXACTLY target's top-level keys and, for each nested
    dict in target, exactly that dict's keys. Any unexpected (extra) key at the
    top level or one level deep fails -- used only for the exact-object metric.
    """
    if not isinstance(got, dict) or set(got.keys()) != set(target.keys()):
        return False
    for k, v in target.items():
        if isinstance(v, dict):
            if not isinstance(got.get(k), dict) or set(got[k].keys()) != set(v.keys()):
                return False
    return True


def score_case(target, got):
    """Return (all_correct, n_correct_fields, n_total_fields, per_field).

    all_correct (the exact-object metric) requires every expected field correct
    AND an exact key set (no unexpected keys). Field accounting is unaffected by
    extra keys -- only expected fields are counted.
    """
    per = {}
    # Flatten one level for field accounting (arguments.* for tool_call).
    flat_exp = {}
    for k, v in target.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat_exp["%s.%s" % (k, kk)] = vv
        else:
            flat_exp[k] = v

    def lookup(dotkey):
        cur = got
        for part in dotkey.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return (False, None)
            cur = cur[part]
        return (True, cur)

    ncorr = 0
    for k, exp in flat_exp.items():
        present, gv = lookup(k)
        ok = present and val_eq(exp, gv)
        per[k] = ok
        if ok:
            ncorr += 1
    passed = (ncorr == len(flat_exp)) and keys_exact(target, got)
    return (passed, ncorr, len(flat_exp), per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    cases = [json.loads(l) for l in open(args.corpus) if l.strip()]
    results = []
    finish_reasons = {}
    leak_marker = 0
    for c in cases:
        resp = chat(args.base_url, args.model, build_messages(c), args.max_tokens)
        choice = resp["choices"][0]
        content = choice["message"]["content"] or ""
        fr = choice.get("finish_reason")
        finish_reasons[fr] = finish_reasons.get(fr, 0) + 1
        leaked = bool(re.search(r"<\|im_(start|end)\|>", content))
        if leaked:
            leak_marker += 1
        got = extract_json(content)
        if got is None:
            passed, ncorr, ntot, per = (False, 0,
                                        len(_flat(c["target"])), {})
            valid = False
        else:
            valid = True
            passed, ncorr, ntot, per = score_case(c["target"], got)
        results.append({
            "id": c["id"], "category": c["category"], "finish_reason": fr,
            "leaked_marker": leaked, "valid_json": valid, "passed": passed,
            "fields_correct": ncorr, "fields_total": ntot,
            "raw": content, "parsed": got, "target": c["target"],
            "per_field": per,
        })

    n = len(results)
    exact = sum(r["passed"] for r in results)
    fc = sum(r["fields_correct"] for r in results)
    ft = sum(r["fields_total"] for r in results)
    valid = sum(r["valid_json"] for r in results)
    by_cat = {}
    for r in results:
        d = by_cat.setdefault(r["category"], {"n": 0, "pass": 0})
        d["n"] += 1
        d["pass"] += r["passed"]
    summary = {
        "n_cases": n,
        "exact_object_pass": exact,
        "exact_object_pass_rate": round(exact / n, 4),
        "valid_json_rate": round(valid / n, 4),
        "field_accuracy": round(fc / ft, 4),
        "fields_correct": fc, "fields_total": ft,
        "finish_reasons": finish_reasons,
        "terminator_leak_count": leak_marker,
        "by_category": {k: {"pass": v["pass"], "n": v["n"]} for k, v in sorted(by_cat.items())},
        "model": args.model, "corpus": args.corpus, "max_tokens": args.max_tokens,
    }
    json.dump({"summary": summary, "results": results}, open(args.out, "w"), indent=2)
    print(json.dumps(summary, indent=2))
    # Loud terminator-health line.
    bad_fr = {k: v for k, v in finish_reasons.items() if k not in ("stop",)}
    if bad_fr or leak_marker:
        print("TERMINATOR-HEALTH WARNING: non-stop finish_reasons=%s leaks=%d"
              % (bad_fr, leak_marker), file=sys.stderr)
    else:
        print("TERMINATOR-HEALTH OK: all %d responses finish_reason=stop, 0 marker leaks"
              % n, file=sys.stderr)


def _flat(target):
    flat = {}
    for k, v in target.items():
        if isinstance(v, dict):
            for kk in v:
                flat["%s.%s" % (k, kk)] = 1
        else:
            flat[k] = 1
    return flat


if __name__ == "__main__":
    main()
