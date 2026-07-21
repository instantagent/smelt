#!/usr/bin/env bash
set -euo pipefail

# Builds libllguidance.a from source (requires Rust) and wraps it into
# CLLGuidance.xcframework. Package.swift prefers this local xcframework over
# the prebuilt GitHub-release artifact, so most users never run this script —
# `swift build` downloads the prebuilt binary instead.
#
# Applies tools/llguidance-serialize.patch on top of the upstream tag: it adds
# TokTrie::to_bytes/from_bytes and SlicedBiasComputer::to_bytes/from_bytes plus
# the llg_tokenizer_to_bytes / llg_new_tokenizer_from_bytes / llg_free_bytes
# FFI, which smelt uses to bake llguidance's tokenizer state (token trie +
# built slicer) into packages (see SmeltLLGuidanceTokenizer serializedTrie).
#
# --package: additionally zip the xcframework and print the checksum to pin in
# Package.swift, plus the upload command for the GitHub release.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REF="${LLG_REF:-v1.7.4}"
PATCH="$ROOT/tools/llguidance-serialize.patch"
SRC="$ROOT/third_party/llguidance/src/llguidance"
OUT="$ROOT/third_party/llguidance/lib"
XCF="$ROOT/third_party/llguidance/CLLGuidance.xcframework"
HEADERS="$ROOT/Sources/CLLGuidance/include"
STAMP="$OUT/.ref"
WANT="$REF+$(shasum -a 256 "$PATCH" | cut -d' ' -f1)"

PACKAGE=0
[[ "${1:-}" == "--package" ]] && PACKAGE=1

mkdir -p "$(dirname "$SRC")" "$OUT"

if [[ ! -f "$OUT/libllguidance.a" || ! -f "$STAMP" || "$(cat "$STAMP")" != "$WANT" ]]; then
  if ! command -v cargo >/dev/null 2>&1; then
    echo "error: cargo not found on PATH" >&2
    echo "  install Rust from https://rustup.rs (or 'brew install rustup-init && rustup-init -y')" >&2
    exit 1
  fi

  if [[ ! -d "$SRC/.git" ]]; then
    git clone https://github.com/guidance-ai/llguidance.git "$SRC"
  fi

  git -C "$SRC" fetch --tags --quiet
  git -C "$SRC" checkout --quiet --force "$REF"
  git -C "$SRC" apply "$PATCH"
  cargo build --manifest-path "$SRC/Cargo.toml" --release -p llguidance
  cp "$SRC/target/release/libllguidance.a" "$OUT/libllguidance.a"
  printf '%s' "$WANT" > "$STAMP.tmp" && mv "$STAMP.tmp" "$STAMP"
fi

if [[ ! -d "$XCF" || "$OUT/libllguidance.a" -nt "$XCF" \
   || "$HEADERS/llguidance.h" -nt "$XCF" || "$HEADERS/module.modulemap" -nt "$XCF" ]]; then
  rm -rf "$XCF"
  xcodebuild -create-xcframework \
    -library "$OUT/libllguidance.a" \
    -headers "$HEADERS" \
    -output "$XCF"
  echo "Built $XCF from llguidance $REF"
  echo "note: SwiftPM caches the evaluated manifest; run 'swift package purge-cache'"
  echo "      if a previous build already resolved the prebuilt URL artifact."
fi

if [[ "$PACKAGE" == 1 ]]; then
  ZIP="$ROOT/third_party/llguidance/CLLGuidance.xcframework.zip"
  rm -f "$ZIP"
  ditto -c -k --keepParent "$XCF" "$ZIP"
  CHECKSUM="$(cd "$ROOT" && swift package compute-checksum "$ZIP")"
  echo "Artifact: $ZIP"
  echo "Checksum: $CHECKSUM"
  echo
  echo "1. Pin the checksum in the CLLGuidance binaryTarget in Package.swift."
  echo "2. Upload this exact zip to the release tag referenced there"
  echo "   (-agentN suffix = revision of tools/llguidance-serialize.patch):"
  echo "   gh release create llguidance-$REF-agent1 \"$ZIP\" --repo smelt-org/binaries"
fi
