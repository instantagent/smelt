// MatvecNoBypassLintTests — the no-bypass LINT, the keystone enforcement of the
// dtype-building-blocks plan (docs/dtype-building-blocks-plan.md §0.5 mech-2, §0.6).
//
// THE INVARIANT IT LOCKS: a matvec kernel pipeline may be NAMED (in a `pipeline:` emission or
// returned by a planner) ONLY inside an allowlisted lowering/planner helper. Combined with the
// gateway (MatvecKernelTable.select, no silent default) + the parity gate (MatvecKernelTableTests),
// this makes "only select() picks a matvec family, and a matvec kernel is emitted only through a
// gateway-routed helper" a STRUCTURAL invariant, not discipline — a new consumer that
// hand-constructs `SmeltDispatch(pipeline: .fp16Matvec)`, aliases the case, or branches on
// `weightEntry.dtype` to pick a matvec kernel FAILS CI.
//
// WHY STRUCTURAL, NOT GREP (plan §0.6): raw grep false-positives on comments, string literals and
// the catalog registry, and misses aliases. This scanner is a small lexer that NEUTRALIZES
// comments + string literals, then brace-matches each `func` to its body so every matvec-pipeline
// reference is attributed to its INNERMOST enclosing function — exact scoping without an AST
// dependency (the project is deliberately dependency-free). Non-vacuity is PROVEN: the scanner has
// a self-test, and negative fixtures feed the linter rogue snippets and assert it flags them.

import Foundation
import Testing

@testable import SmeltCompiler

// MARK: - The source scanner (comment/string-neutralizing lexer + function attribution)

private enum SwiftSourceScanner {
    /// Replace the CONTENTS of // line comments, /* */ block comments, and "..."/"""...""" string
    /// literals with spaces (preserving length + newlines), so brace-matching and token scanning
    /// over the result can't be fooled by braces or pipeline-case names inside comments/strings.
    static func neutralize(_ source: String) -> [Character] {
        var out = Array(source)
        let n = out.count
        var i = 0
        func blank(_ from: Int, _ to: Int) {
            var k = from
            while k < to { if out[k] != "\n" { out[k] = " " }; k += 1 }
        }
        while i < n {
            let c = out[i]
            // Line comment.
            if c == "/", i + 1 < n, out[i + 1] == "/" {
                var j = i
                while j < n, out[j] != "\n" { j += 1 }
                blank(i, j); i = j; continue
            }
            // Block comment — Swift allows NESTING (/* /* */ */), so depth-count to the balanced
            // close, not the first */ (else a nested comment could leak fake code to the scanner).
            if c == "/", i + 1 < n, out[i + 1] == "*" {
                var j = i + 2
                var depth = 1
                while j + 1 < n, depth > 0 {
                    if out[j] == "/", out[j + 1] == "*" { depth += 1; j += 2 }
                    else if out[j] == "*", out[j + 1] == "/" { depth -= 1; j += 2 }
                    else { j += 1 }
                }
                let end = min(j, n)
                blank(i, end); i = end; continue
            }
            // Raw string literal: #"..."#, ##"..."##, #"""..."""# (the # count balances the close).
            // Raw strings may contain unescaped " and braces, so they must be matched by their
            // hash-balanced closer, not a bare ".
            if c == "#" {
                var hashes = 0
                var p = i
                while p < n, out[p] == "#" { hashes += 1; p += 1 }
                if p < n, out[p] == "\"" {
                    let triple = p + 2 < n && out[p + 1] == "\"" && out[p + 2] == "\""
                    let openLen = triple ? 3 : 1
                    let closer = Array(String(repeating: "\"", count: openLen)
                        + String(repeating: "#", count: hashes))
                    let contentStart = p + openLen
                    var j = contentStart
                    while j + closer.count <= n {
                        var m = true
                        for t in 0..<closer.count where out[j + t] != closer[t] { m = false; break }
                        if m { break }
                        j += 1
                    }
                    blank(contentStart, j)
                    i = min(n, j + closer.count); continue
                }
            }
            // Multiline string.
            if c == "\"", i + 2 < n, out[i + 1] == "\"", out[i + 2] == "\"" {
                var j = i + 3
                while j + 2 < n, !(out[j] == "\"" && out[j + 1] == "\"" && out[j + 2] == "\"") { j += 1 }
                let end = min(j + 3, n)
                blank(i + 3, max(i + 3, end - 3)); i = end; continue
            }
            // Single-line string (with \-escapes).
            if c == "\"" {
                var j = i + 1
                while j < n, out[j] != "\"" {
                    if out[j] == "\\" { j += 2 } else { j += 1 }
                }
                let end = min(j + 1, n)
                blank(i + 1, max(i + 1, end - 1)); i = end; continue
            }
            i += 1
        }
        return out
    }

    struct FuncRegion { let name: String; let bodyStart: Int; let bodyEnd: Int }

    /// Every `func NAME` in the neutralized source, with the char range of its body `{ ... }`.
    /// Bodies nest; a token is attributed to the SMALLEST (innermost) region containing it.
    static func functionRegions(_ chars: [Character]) -> [FuncRegion] {
        let n = chars.count
        var regions: [FuncRegion] = []
        var i = 0
        while i < n {
            // Match the keyword `func` on a word boundary.
            if chars[i] == "f", matchesWord(chars, i, "func") {
                var j = i + 4
                // Skip whitespace, then read the name.
                while j < n, chars[j] == " " || chars[j] == "\n" || chars[j] == "\t" { j += 1 }
                var name = ""
                while j < n, chars[j].isLetter || chars[j].isNumber || chars[j] == "_" {
                    name.append(chars[j]); j += 1
                }
                // Skip the balanced parameter list (...) so a default-closure argument `= { }` is
                // not mistaken for the body brace (codex U1d review #6).
                while j < n, chars[j] != "(", chars[j] != "{" { j += 1 }
                if j < n, chars[j] == "(" {
                    var pd = 0
                    while j < n {
                        if chars[j] == "(" {
                            pd += 1
                        } else if chars[j] == ")" {
                            pd -= 1
                            if pd == 0 { j += 1; break }
                        }
                        j += 1
                    }
                }
                // Find the body's opening brace (first '{' after the signature).
                while j < n, chars[j] != "{" { j += 1 }
                if j < n, !name.isEmpty {
                    let bodyStart = j + 1
                    var depth = 0
                    var k = j
                    while k < n {
                        if chars[k] == "{" { depth += 1 }
                        else if chars[k] == "}" { depth -= 1; if depth == 0 { break } }
                        k += 1
                    }
                    regions.append(FuncRegion(name: name, bodyStart: bodyStart, bodyEnd: k))
                }
                i = max(i + 4, j)
                continue
            }
            i += 1
        }
        return regions
    }

    /// The innermost function whose body contains `index`, or nil if at file scope.
    static func enclosingFunction(_ regions: [FuncRegion], _ index: Int) -> String? {
        var best: FuncRegion?
        for r in regions where index >= r.bodyStart && index < r.bodyEnd {
            if best == nil || (r.bodyEnd - r.bodyStart) < (best!.bodyEnd - best!.bodyStart) {
                best = r
            }
        }
        return best?.name
    }

    private static func matchesWord(_ chars: [Character], _ i: Int, _ word: String) -> Bool {
        let w = Array(word)
        guard i + w.count <= chars.count else { return false }
        for k in 0..<w.count where chars[i + k] != w[k] { return false }
        // Preceding char must not be an identifier char (word boundary).
        if i > 0 {
            let p = chars[i - 1]
            if p.isLetter || p.isNumber || p == "_" || p == "." { return false }
        }
        let after = i + w.count
        if after < chars.count {
            let a = chars[after]
            if a.isLetter || a.isNumber || a == "_" { return false }
        }
        return true
    }

    /// Every `.<caseName>` member-access in the neutralized source whose caseName is in `cases`,
    /// returned as (caseName, charIndex). Matches both `.case` and `Type.case` forms (the `.`).
    static func memberAccesses(_ chars: [Character], cases: Set<String>) -> [(name: String, index: Int)] {
        let n = chars.count
        var hits: [(String, Int)] = []
        var i = 0
        while i < n {
            if chars[i] == "." {
                var j = i + 1
                var name = ""
                while j < n, chars[j].isLetter || chars[j].isNumber || chars[j] == "_" {
                    name.append(chars[j]); j += 1
                }
                if cases.contains(name) { hits.append((name, i)) }
                i = max(i + 1, j)
                continue
            }
            i += 1
        }
        return hits
    }
}

// MARK: - The matvec pipeline-case set (derived from the enum, not hardcoded)

/// A SmeltPipeline case is a MATVEC kernel iff it performs a weight×input matmul — NOT a pure
/// activation (swigluFused), norm, embedding gather, conv, rope, or swap. Classified by name so
/// the set tracks the enum automatically; a sanity test pins known in/out members so a misclassification
/// can't silently shrink the guarded set (anti-theater).
private func isMatvecPipelineName(_ name: String) -> Bool {
    let n = name.lowercased()
    if n.contains("embedding") || n.contains("gather") { return false }
    return n.contains("matvec") || n.contains("matmul")
        || n.hasPrefix("gemv") || n.hasPrefix("gemm")
        || n.contains("gateup")   // fused gate+up projection (matmuls gate & up)
        || n.contains("qkv")      // fused q/k/v projection
}

/// Weight×input dot-product kernels whose NAME carries no matvec/gemv token (so the classifier
/// misses them) — pinned explicitly so the guarded set can't silently exclude them. clusterSparse
/// LMHead is a sparse LM-head dot-product over lm_head_weight (dispatched in TopLevelEmitter).
private let extraMatvecCaseNames: Set<String> = ["clusterSparseLMHead"]

private let matvecPipelineCaseNames: Set<String> =
    Set(SmeltPipeline.allCases.map { "\($0)" }
        .filter { isMatvecPipelineName($0) || extraMatvecCaseNames.contains($0) })

// MARK: - The allowlist (role-split, file-qualified)

/// Whole-file allowlist: pure planner/registry files that NAME matvec pipelines (map shapes →
/// specialized pipelines, return routes) but never construct a bypassing dispatch. The catalog is
/// excluded from the scan entirely (it is the registry definition).
/// The matvec-pipeline PRODUCER / REGISTRY / OPTIMIZER subsystem (a documented TRUST BOUNDARY):
/// these files NAME, map, route, classify, and construct matvec dispatches as their CORE JOB —
/// they are the implementation BEHIND the gateway, not consumers that could bypass select(). The
/// consumer surface (the emitters that USE these) is what the lint enforces: invariant (a)/(b)
/// scope the emitter files, and invariant (c) forbids any consumer from getting a matvec route out
/// of SmeltFusionPlanner outside a gateway-routed helper. Function-level allowlisting these would
/// require listing every shape→pipeline route-table accessor; the trust boundary is the honester,
/// reviewable line.
private let allowlistedPlannerFiles: Set<String> = [
    "SmeltFusionPlanner.swift",          // shape → specialized matvec route (+ constructs them)
    "SmeltKernelShapeRegistry.swift",    // shape → pipeline maps
    "SmeltDispatchOptimizer.swift",      // classifies + rewrites/fuses recorded matvec dispatches
    "SmeltRigPackageBuilder.swift",      // rig package pipeline inventory (kept DRY w/ the catalog)
    "Qwen3TTSPackageBuilder.swift",      // ttsPipelines DECLARATION (kept DRY w/ the catalog)
]

/// Function-level, FILE-QUALIFIED allowlist for the EMITTER files: the lowering helpers (emit a
/// matvec pipeline AFTER the gateway picked the family) + the gateway selectors. A matvec pipeline
/// named in any OTHER function (or at file scope) of these files is a bypass. File-qualified so a
/// generic name (`generate`, `gemm`) is blessed only in the emitter where it legitimately lowers.
private let allowlistedFunctions: Set<String> = [
    // SmeltCodeEmitter: gateway selectors + committed lowerers.
    "SmeltCodeEmitter.swift:matvecFamily",
    "SmeltCodeEmitter.swift:optionalFusedFamily",
    "SmeltCodeEmitter.swift:bothFusedFamily",
    "SmeltCodeEmitter.swift:fp16DenseMatvecPipeline",  // dense weight dtype → fp16-act matvec kernel
    "SmeltCodeEmitter.swift:emitMatvec",
    "SmeltCodeEmitter.swift:emitMatvecVar",
    "SmeltCodeEmitter.swift:emitFP16Matvec",
    "SmeltCodeEmitter.swift:emitFP16MatvecFP32Out",
    "SmeltCodeEmitter.swift:emitFP16MatvecVar",
    "SmeltCodeEmitter.swift:emitAffineMatvec",
    "SmeltCodeEmitter.swift:emitAffineMatvecImpl",
    "SmeltCodeEmitter.swift:emitAffineMatvecVar",
    "SmeltCodeEmitter.swift:emitFusedLUTMatvec",
    "SmeltCodeEmitter.swift:emitFusedLUTMatvecVar",
    "SmeltCodeEmitter.swift:emitTQHMatvec",
    "SmeltCodeEmitter.swift:emitFusedGateUpSwiglu",
    "SmeltCodeEmitter.swift:emitFusedAffineGateUpSwiglu",
    "SmeltCodeEmitter.swift:emitFusedAffineGateUpGeGLU",
    "SmeltCodeEmitter.swift:emitFusedDualLutMatvec",
    "SmeltCodeEmitter.swift:emitFusedDualAffineMatvec",
    "SmeltCodeEmitter.swift:emitFusedDualAffineMatvecAdd",
    "SmeltCodeEmitter.swift:emitNormScaleAffineMatvecIfPossible",
    "SmeltCodeEmitter.swift:emitNormScaleAffineGateUpGeGLUIfPossible",
    // Signed-storage and CAM-owned activation-view lowerers. These functions
    // are the reviewed family boundary for binary/ternary graph views: they
    // validate storage, view, and exact geometry before naming a pipeline.
    "SmeltCodeEmitter.swift:emitSignedMatvec",
    "SmeltCodeEmitter.swift:emitSignedPackedProjectionBankIfPossible",
    "SmeltCodeEmitter.swift:emitSignedBitplaneProjectionBankIfPossible",
    "SmeltCodeEmitter.swift:emitSignedBitplaneMatvecIfPossible",
    "SmeltCodeEmitter.swift:emitSignedBitplaneMatmulBatchedIfPossible",
    "SmeltCodeEmitter.swift:emitSignedBinaryGateUpSwiglu",
    "SmeltCodeEmitter.swift:emitSignedTernaryAffineGateUpSwiglu",
    "SmeltCodeEmitter.swift:emitSignedBitplaneGateUpSwigluIfPossible",
    // PrefillEmitter: lowerers / routed helpers.
    "PrefillEmitter.swift:emitBatchedMatmul",
    "PrefillEmitter.swift:emitBatchedAffineMatmul",
    "PrefillEmitter.swift:emitBatchedFP16MatvecFP32Out",
    "PrefillEmitter.swift:emitMatvecVarSlice",
    "PrefillEmitter.swift:emitMatvecFixed",
    "PrefillEmitter.swift:emitUnrolledFusedDualAffineMatvec",
    "PrefillEmitter.swift:emitUnrolledFusedAffineGateUpSwiglu",
    "PrefillEmitter.swift:emitLMHeadLastToken",
    "PrefillEmitter.swift:emitLMHeadAtPosition",
    "PrefillEmitter.swift:emitVerifyArgmaxLMHead",
    // DenseTrunkEmitter (talker decode): generate emits the fused qkv/gateup inline; nested helpers.
    "DenseTrunkEmitter.swift:generate",
    "DenseTrunkEmitter.swift:gemvAdd",
    "DenseTrunkEmitter.swift:emitProjGemv",
    "DenseTrunkEmitter.swift:emitGemvDense",
    "DenseTrunkEmitter.swift:emitGemvU4",
    // DenseTrunkPrefillEmitter (talker prefill): nested gemm/proj helpers.
    "DenseTrunkPrefillEmitter.swift:gemm",
    "DenseTrunkPrefillEmitter.swift:emitProjGemv",
    "DenseTrunkPrefillEmitter.swift:emitGemvDense",
    "DenseTrunkPrefillEmitter.swift:emitGemvU4",
    // Orchestrators that route through the gateway (bothFusedFamily / gateUpFamily) BEFORE calling
    // a matvec route provider or emitting the LM head — gateway-routed, not leaf lowerers.
    "TopLevelEmitter.swift:generatePlanned",                       // emits the clusterSparseLMHead
    "PrefillEmitter.swift:emitBatchedFFN",                         // prefillFusedGateUpFull route
    "PrefillEmitter.swift:emitPerLayerResidualBranchBatched", // prefillAffineBatched route
]

// MARK: - Reading the production sources

private func sourceDir() -> URL {
    // This file lives at Tests/SmeltCompilerTests/; the sources are at Sources/SmeltCompiler/.
    URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Sources/SmeltCompiler")
}

/// Every .swift under Sources/SmeltCompiler RECURSIVELY (a future nested subdirectory must not be
/// invisible to the lint), excluding the catalog registry.
private func swiftSourcesRecursively() -> [URL] {
    let dir = sourceDir()
    guard let en = FileManager.default.enumerator(at: dir, includingPropertiesForKeys: nil) else {
        return []
    }
    var out: [URL] = []
    for case let url as URL in en
    where url.pathExtension == "swift" && url.lastPathComponent != "SmeltKernelCatalog.swift" {
        out.append(url)
    }
    return out
}

/// One production source read + neutralized + brace-parsed ONCE, reused by all three invariant
/// scans (and the guarded-ref count) instead of re-lexing the same file per invariant.
private struct ScannedSource {
    let fileName: String
    let chars: [Character]
    let regions: [SwiftSourceScanner.FuncRegion]
    init(fileName: String, text: String) {
        self.fileName = fileName
        self.chars = SwiftSourceScanner.neutralize(text)
        self.regions = SwiftSourceScanner.functionRegions(chars)
    }
}

private func scannedProductionSources() -> [ScannedSource] {
    swiftSourcesRecursively().compactMap { url in
        (try? String(contentsOf: url, encoding: .utf8))
            .map { ScannedSource(fileName: url.lastPathComponent, text: $0) }
    }
}

private struct Bypass: CustomStringConvertible {
    let file: String, caseName: String, function: String?
    var description: String {
        "\(file): matvec pipeline .\(caseName) named in "
            + (function.map { "non-allowlisted function \($0)()" } ?? "file scope")
    }
}

/// Scan one file's text for matvec pipeline references outside the allowlist. `fileName` selects
/// whole-file allowlisting; otherwise every reference must be in an allowlisted function.
private func bypassesIn(_ s: ScannedSource) -> [Bypass] {
    if allowlistedPlannerFiles.contains(s.fileName) { return [] }
    var found: [Bypass] = []
    for hit in SwiftSourceScanner.memberAccesses(s.chars, cases: matvecPipelineCaseNames) {
        let fn = SwiftSourceScanner.enclosingFunction(s.regions, hit.index)
        if fn == nil || !allowlistedFunctions.contains("\(s.fileName):\(fn!)") {
            found.append(Bypass(file: s.fileName, caseName: hit.name, function: fn))
        }
    }
    return found
}

// MARK: - Scanner self-test (trust the scanner before trusting the scan)

@Test func scannerNeutralizesAndAttributesCorrectly() {
    let snippet = """
    struct E {
        // a comment with .fp16Matvec must be ignored
        func allowed() {
            let s = "a string with .fp16Matvec and a stray { brace"
            emit(pipeline: .fp16Matvec)   // real reference, inside allowed()
        }
        func helper() {
            func nested() { let p = SmeltPipeline.gemvF32 }
        }
    }
    """
    let chars = SwiftSourceScanner.neutralize(snippet)
    let regions = SwiftSourceScanner.functionRegions(chars)
    let hits = SwiftSourceScanner.memberAccesses(chars, cases: ["fp16Matvec", "gemvF32"])
    // The comment + string occurrences are neutralized → exactly two real references.
    #expect(hits.count == 2)
    // Raw strings (with inner " and braces) must also be neutralized — not break brace-matching.
    let raw = """
    struct R {
        func r() {
            let p = #"a raw "string" with .fp16Matvec and { unbalanced braces"#
            let q = .gemvF32
        }
    }
    """
    let rc = SwiftSourceScanner.neutralize(raw)
    let rr = SwiftSourceScanner.functionRegions(rc)
    let rh = SwiftSourceScanner.memberAccesses(rc, cases: ["fp16Matvec", "gemvF32"])
    #expect(rh.count == 1)   // only .gemvF32 — the raw string's .fp16Matvec is neutralized
    #expect(SwiftSourceScanner.enclosingFunction(rr, rh[0].index) == "r")  // brace-match survived
    // A default-closure parameter `= { }` must not be mistaken for the body brace.
    let cd = """
    struct C {
        func f(cb: () -> Void = { let z = 1 }) {
            let p = .gemvF32
        }
    }
    """
    let cc = SwiftSourceScanner.neutralize(cd)
    let cr = SwiftSourceScanner.functionRegions(cc)
    let ch = SwiftSourceScanner.memberAccesses(cc, cases: ["gemvF32"])
    #expect(ch.count == 1)
    #expect(SwiftSourceScanner.enclosingFunction(cr, ch[0].index) == "f")
    let owners = Set(hits.map { SwiftSourceScanner.enclosingFunction(regions, $0.index) })
    #expect(owners == Set(["allowed", "nested"]))   // innermost attribution (nested, not helper)
}

// MARK: - The matvec-case set is sane (non-vacuity of the guarded set)

@Test func matvecPipelineSetIsSaneAndNonEmpty() {
    // Known matvec kernels MUST be in the set...
    for c in ["fp16Matvec", "affineMatvec", "gemvF32", "gemmBF16WF32", "gemvU4F32",
              "fusedLutMatvec", "fusedAffineGateUpSwiglu", "fusedDualLutMatvec",
              "gemvQKVBF16WF32", "tqhMatvec", "clusterSparseLMHead"] {
        #expect(matvecPipelineCaseNames.contains(c), "expected matvec: \(c)")
    }
    // ...and known NON-matvec kernels must NOT be (or the guard would over/under-reach).
    for c in ["swigluFused", "swigluF32", "rmsNorm1PW", "embeddingGather", "snakeActivation"] {
        #expect(!matvecPipelineCaseNames.contains(c), "must NOT be matvec: \(c)")
    }
    #expect(matvecPipelineCaseNames.count >= 20)
}

// MARK: - The invariant: no bypass in the real production sources

@Test func noMatvecPipelineBypassInProductionSources() throws {
    let sources = scannedProductionSources()
    #expect(sources.count >= 10, "expected to scan the SmeltCompiler sources, found \(sources.count)")

    var allBypasses: [Bypass] = []
    var guardedRefs = 0
    for s in sources {
        allBypasses += bypassesIn(s)
        // Count guarded (allowlisted) references for the non-vacuity floor — same scan, no re-lex.
        if !allowlistedPlannerFiles.contains(s.fileName) {
            for hit in SwiftSourceScanner.memberAccesses(s.chars, cases: matvecPipelineCaseNames) {
                if let fn = SwiftSourceScanner.enclosingFunction(s.regions, hit.index),
                   allowlistedFunctions.contains("\(s.fileName):\(fn)") { guardedRefs += 1 }
            }
        }
    }
    let report = allBypasses.map(\.description).joined(separator: "\n")
    #expect(allBypasses.isEmpty, "matvec no-bypass violations:\n\(report)")
    // Non-vacuity: the scan actually saw matvec emissions inside allowlisted helpers.
    #expect(guardedRefs >= 15, "expected the scan to find guarded matvec emissions, got \(guardedRefs)")
}

// MARK: - Invariant (b): no dtype-routing to matvec emitters outside the gateway

/// Names whose call (`name(`) means "emit a matvec kernel". A QUANT-dtype comparison co-located
/// with one of these is the family-selection-by-hand bypass that pipeline-emission scoping alone
/// does not catch (the caller names no pipeline; the pipeline lives inside the called helper).
private let matvecEmitterCallNames: [String] = [
    "emitMatvec", "emitMatvecVar", "emitMatvecVarSlice", "emitMatvecFixed",
    "emitFP16Matvec", "emitFP16MatvecVar", "emitFP16MatvecFP32Out",
    "emitAffineMatvec", "emitAffineMatvecVar",
    "emitFusedLUTMatvec", "emitFusedLUTMatvecVar", "emitTQHMatvec",
    "emitBatchedMatmul", "emitBatchedAffineMatmul", "emitBatchedFP16MatvecFP32Out",
    "emitFusedGateUpSwiglu", "emitFusedAffineGateUpSwiglu", "emitFusedAffineGateUpGeGLU",
    "emitFusedDualLutMatvec", "emitFusedDualAffineMatvec",
]

/// A function does QUANT-dtype routing if it compares a weight's dtype to a QUANT case — the
/// family decision the gateway owns. We deliberately do NOT flag `.dtype == .fp16/.bf16/.fp32`:
/// those are legit OUTPUT-variant / validation decisions (e.g. the fp32-out down-proj), the
/// activation/output axis a later unit routes — not matvec FAMILY selection.
// Compiled ONCE (hoisted out of functionDoesQuantDtypeRouting, which runs per function body).
private let quantDtypeEqRegex = try! NSRegularExpression(
    pattern: #"(\w+)\.dtype\s*==\s*\.(affineU4|u4Lut|turboQuantH)"#)
private let quantCaseRegex = try! NSRegularExpression(
    pattern: #"case\s+\.(affineU4|u4Lut|turboQuantH)\b"#)
private let switchDtypeRegex = try! NSRegularExpression(
    pattern: #"switch\s+[\w.]*?(\w+)\.dtype"#)

private func functionDoesQuantDtypeRouting(_ body: String) -> Bool {
    // A function routes a matvec on a QUANT dtype if it compares (or switches) a matvec WEIGHT
    // entry's dtype against a quant case. We exclude embedding entries (embedEntry/perLayerEmbed
    // Entry quant routing picks an embedding-gather kernel — the U4 EMBEDDING axis a later unit
    // owns — not a matvec family) and dense `.fp16/.bf16/.fp32` (output-variant / validation).
    let ns = body as NSString
    let full = NSRange(location: 0, length: ns.length)
    func varIsWeight(_ name: String) -> Bool { !name.lowercased().contains("embed") }
    // Form 1: `<weightVar>.dtype == .<quant>` (if/guard comparison).
    for m in quantDtypeEqRegex.matches(in: body, range: full)
    where varIsWeight(ns.substring(with: m.range(at: 1))) { return true }
    // Form 2: `switch <weightVar>.dtype { ... case .<quant>: ... }` — a switch-based router (no
    // `==`, so Form 1 misses it). Trips only if the body also matches a quant `case`.
    if quantCaseRegex.firstMatch(in: body, range: full) != nil {
        for m in switchDtypeRegex.matches(in: body, range: full)
        where varIsWeight(ns.substring(with: m.range(at: 1))) { return true }
    }
    return false
}

private func functionCallsMatvecEmitter(_ body: String) -> Bool {
    matvecEmitterCallNames.contains { body.contains("\($0)(") }
}

/// Flag any NON-allowlisted function that BOTH compares a quant dtype AND calls a matvec emitter
/// — i.e. picks a matvec kernel by hand-routing on dtype instead of going through select().
private func dtypeRoutingBypassesIn(_ s: ScannedSource) -> [Bypass] {
    if allowlistedPlannerFiles.contains(s.fileName) { return [] }
    var found: [Bypass] = []
    for r in s.regions
    where !allowlistedFunctions.contains("\(s.fileName):\(r.name)") {
        let body = String(s.chars[r.bodyStart..<r.bodyEnd])
        if functionDoesQuantDtypeRouting(body) && functionCallsMatvecEmitter(body) {
            found.append(Bypass(file: s.fileName, caseName: "dtype-routing", function: r.name))
        }
    }
    return found
}

@Test func noQuantDtypeRoutingToMatvecOutsideTheGateway() throws {
    var bypasses: [Bypass] = []
    for s in scannedProductionSources() {
        bypasses += dtypeRoutingBypassesIn(s)
    }
    let report = bypasses.map { "\($0.file): \($0.function ?? "?")() routes a quant dtype to a matvec emitter" }
        .joined(separator: "\n")
    #expect(bypasses.isEmpty, "quant-dtype matvec-routing bypasses:\n\(report)")
}

@Test func lintFlagsQuantDtypeRoutingBypass() {
    let rogue = """
    struct X {
        func rogue(_ e: SmeltCodeEmitter) {
            if weightEntry.dtype == .affineU4 {
                _ = try e.emitAffineMatvec()
            } else {
                _ = try e.emitFP16Matvec()
            }
        }
    }
    """
    #expect(!dtypeRoutingBypassesIn(ScannedSource(fileName: "X.swift", text: rogue)).isEmpty)
}

@Test func lintFlagsSwitchDtypeRoutingBypass() {
    // The switch-based router has no `==` — Form 2 of functionDoesQuantDtypeRouting must catch it.
    let rogue = """
    struct X {
        func rogue(_ e: SmeltCodeEmitter) {
            switch weightEntry.dtype {
            case .affineU4: _ = try e.emitAffineMatvec()
            default: _ = try e.emitFP16Matvec()
            }
        }
    }
    """
    #expect(!dtypeRoutingBypassesIn(ScannedSource(fileName: "X.swift", text: rogue)).isEmpty)
}

@Test func lintAllowsFP16OutputVariantDecision() {
    // The fp32-out down-proj decision routes on `.dtype == .fp16` (an OUTPUT variant), NOT a quant
    // family — it must NOT be flagged as a family-selection bypass.
    let ok = """
    struct X {
        func emitFFN(_ e: SmeltCodeEmitter) {
            if downEntry.dtype == .fp16 {
                _ = try e.emitFP16MatvecFP32Out()
            } else {
                _ = try e.emitBatchedMatmul()
            }
        }
    }
    """
    #expect(dtypeRoutingBypassesIn(ScannedSource(fileName: "TopLevelEmitter.swift", text: ok)).isEmpty)
}

// MARK: - Invariant (c): matvec ROUTE PROVIDERS used only inside gateway-routed helpers

/// Functions that RETURN a matvec pipeline/route without the caller naming a case or comparing a
/// dtype. Getting one and emitting `SmeltDispatch(pipeline: <returned>)` is a bypass invisible to
/// (a)+(b). So they may be CALLED only from allowlisted (gateway-routed) helpers. Two kinds:
///   - SmeltFusionPlanner route methods (return a struct carrying a specialized matvec pipeline).
///   - fp16DenseMatvecPipeline: the dense weight-dtype → fp16-act matvec PIPELINE selector. It is
///     gateway-adjacent (callers reach it only after select() yields `.dense(dt)`), but a future
///     non-allowlisted helper could call it and alias the result — so it is route-scoped here too.
private let matvecRouteProviders: [String] = [
    "decodeAffineMatvec", "decodeDualAffineMatvec", "decodeFusedGeGLU", "decodeFusedSwiGLU",
    "decodeNormScaleAffine", "decodeNormScaleGeGLU",
    "prefillAffineBatched", "prefillAffineFull", "prefillDualAffineMatvec",
    "prefillFusedGateUpFull", "unrolledPrefillAffineMatvec",
    "fp16DenseMatvecPipeline",
]

private func routeProviderBypassesIn(_ src: ScannedSource) -> [Bypass] {
    if allowlistedPlannerFiles.contains(src.fileName) { return [] }
    let text = String(src.chars)
    var found: [Bypass] = []
    for provider in matvecRouteProviders {
        let needle = ".\(provider)("
        var idx = text.startIndex
        while let r = text.range(of: needle, range: idx..<text.endIndex) {
            let charIndex = text.distance(from: text.startIndex, to: r.lowerBound)
            let fn = SwiftSourceScanner.enclosingFunction(src.regions, charIndex)
            if fn == nil || !allowlistedFunctions.contains("\(src.fileName):\(fn!)") {
                found.append(Bypass(file: src.fileName, caseName: "route:\(provider)", function: fn))
            }
            idx = r.upperBound
        }
    }
    return found
}

@Test func noMatvecRouteProviderUseOutsideTheGateway() throws {
    var bypasses: [Bypass] = []
    for s in scannedProductionSources() {
        bypasses += routeProviderBypassesIn(s)
    }
    let report = bypasses.map { "\($0.file): \($0.function ?? "file scope")() uses \($0.caseName)" }
        .joined(separator: "\n")
    #expect(bypasses.isEmpty, "matvec route-provider used outside a gateway-routed helper:\n\(report)")
}

@Test func lintFlagsAliasedRouteEmission() {
    // The aliased-emission bypass: get a matvec route from the planner, emit its pipeline. No case
    // name, no dtype compare — only the route-provider CALL betrays it.
    let rogue = """
    struct X {
        func rogue(_ e: SmeltCodeEmitter) {
            let route = e.fusionPlanner.prefillAffineBatched(rows: 1, cols: 1, groupSize: 1)
            _ = try e.emit(SmeltDispatch(pipeline: route.pipeline))
        }
    }
    """
    #expect(!routeProviderBypassesIn(ScannedSource(fileName: "X.swift", text: rogue)).isEmpty)
}

@Test func lintFlagsAliasedDensePipelineSelector() {
    // The fp16DenseMatvecPipeline dodge: a non-allowlisted helper calls the dense selector and
    // emits the returned pipeline — names no case, compares no QUANT dtype, so (a)+(b) miss it.
    // Route-scoping the selector (invariant c) catches the CALL.
    let rogue = """
    struct X {
        func rogue(_ e: SmeltCodeEmitter, _ dt: SmeltDType) {
            let p = try SmeltCodeEmitter.fp16DenseMatvecPipeline(dt)
            _ = try e.emit(SmeltDispatch(pipeline: p))
        }
    }
    """
    #expect(!routeProviderBypassesIn(ScannedSource(fileName: "X.swift", text: rogue)).isEmpty)
}

// MARK: - Negative fixtures: the linter MUST flag known bypasses (anti-theater)

@Test func lintFlagsRogueDirectEmission() {
    let rogue = """
    struct X {
        func rogue() {
            _ = SmeltDispatch(pipeline: .fp16Matvec)
        }
    }
    """
    #expect(!bypassesIn(ScannedSource(fileName: "X.swift", text: rogue)).isEmpty)
}

@Test func lintFlagsAliasedRoguePipeline() {
    let rogue = """
    struct X {
        func rogue() {
            let p = SmeltPipeline.gemvU4F32
            _ = SmeltDispatch(pipeline: p)
        }
    }
    """
    // The aliasing line `SmeltPipeline.gemvU4F32` is itself a matvec member-access in a
    // non-allowlisted function — caught even though `pipeline: p` hides the case.
    #expect(!bypassesIn(ScannedSource(fileName: "X.swift", text: rogue)).isEmpty)
}

@Test func lintFlagsBypassInAnAllowlistedEmitterFileButWrongFunction() {
    // A matvec pipeline named in a NEW (non-allowlisted) function of an emitter file is a bypass,
    // even though other functions in that file are allowlisted.
    let rogue = """
    extension SmeltCodeEmitter {
        func newHelperThatBypasses() {
            _ = SmeltDispatch(pipeline: .affineMatvec)
        }
    }
    """
    #expect(!bypassesIn(ScannedSource(fileName: "SmeltCodeEmitter.swift", text: rogue)).isEmpty)
}

@Test func lintAllowsMatvecInsideAnAllowlistedFunction() {
    let ok = """
    struct E {
        func emitFP16Matvec() {
            _ = SmeltDispatch(pipeline: .fp16Matvec)
        }
    }
    """
    #expect(bypassesIn(ScannedSource(fileName: "SmeltCodeEmitter.swift", text: ok)).isEmpty)
}
