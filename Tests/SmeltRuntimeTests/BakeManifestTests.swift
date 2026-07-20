// U1 gate for the bake-parity arc: the `baked.json` honesty marker
// (SmeltBakeManifest) — round-trip, recording semantics, backfill of a legacy
// package, CAS-symlink survival, and single-source filename consistency.
// No runtime behavior change yet (enforcement is U2a/b/c).

import XCTest
import SmeltSchema
@testable import SmeltRuntime

final class BakeManifestTests: XCTestCase {

    private func makeTempPackage() throws -> String {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("baketest-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
        return dir.path
    }

    private func touch(_ packagePath: String, _ name: String, _ body: String = "{}") throws {
        try body.write(
            toFile: "\(packagePath)/\(name)", atomically: true, encoding: .utf8)
    }

    private func writeDescriptorManifest(
        _ packagePath: String,
        label: String,
        architecture: String,
        blocks: SmeltBlockGraph
    ) throws {
        struct DescriptorManifest: Encodable {
            let kind: String
            let architecture: String
            let blocks: SmeltBlockGraph
        }
        let data = try JSONEncoder().encode(DescriptorManifest(
            kind: label,
            architecture: architecture,
            blocks: blocks
        ))
        try touch(
            packagePath,
            "manifest.json",
            String(decoding: data, as: UTF8.self)
        )
    }

    // MARK: - Round-trip

    func testRoundTrips() throws {
        let pkg = try makeTempPackage()
        let manifest = SmeltBakeManifest(sealed: [
            SmeltBakeManifest.prefix(),
            SmeltBakeManifest.grammar(hasTrie: true),
        ])
        try manifest.write(packagePath: pkg)
        let loaded = try SmeltBakeManifest.load(packagePath: pkg)
        XCTAssertEqual(loaded, manifest)
    }

    func testLoadAbsentIsNil() throws {
        let pkg = try makeTempPackage()
        XCTAssertNil(try SmeltBakeManifest.load(packagePath: pkg))
    }

    func testLoadMalformedThrows() throws {
        let pkg = try makeTempPackage()
        try touch(pkg, SmeltBakeManifest.fileName, "not json")
        XCTAssertThrowsError(try SmeltBakeManifest.load(packagePath: pkg))
    }

    // MARK: - Recording semantics

    func testRecordingReplacesByKindAndSorts() throws {
        let base = SmeltBakeManifest(sealed: [SmeltBakeManifest.voice()])
        let withGrammarNoTrie = base.recording(SmeltBakeManifest.grammar(hasTrie: false))
        // Re-record grammar with a trie: replaces, does not duplicate.
        let withGrammarTrie = withGrammarNoTrie.recording(
            SmeltBakeManifest.grammar(hasTrie: true))
        XCTAssertEqual(withGrammarTrie.sealed.filter { $0.kind == .grammar }.count, 1)
        XCTAssertEqual(
            withGrammarTrie.sealed.first { $0.kind == .grammar }?.perf,
            [SmeltBakeArtifacts.grammarTrie])
        // Deterministic order by raw kind name: grammar < voice.
        XCTAssertEqual(withGrammarTrie.sealed.map(\.kind), [.grammar, .voice])
    }

    // MARK: - record() write + accumulate

    func testRecordCreatesAndAccumulates() throws {
        let pkg = try makeTempPackage()
        try SmeltBakeManifest.record([SmeltBakeManifest.prefix()], packagePath: pkg)
        try SmeltBakeManifest.record([SmeltBakeManifest.grammar(hasTrie: true)], packagePath: pkg)
        let loaded = try XCTUnwrap(try SmeltBakeManifest.load(packagePath: pkg))
        XCTAssertEqual(Set(loaded.sealed.map(\.kind)), [.prefix, .grammar])
    }

    // MARK: - Backfill (migration of a legacy package)

    func testInferExistingFromSidecars() throws {
        let pkg = try makeTempPackage()
        try touch(pkg, SmeltBakeArtifacts.prefixMeta)
        try touch(pkg, SmeltBakeArtifacts.prefixSnapshot)
        try touch(pkg, SmeltBakeArtifacts.grammarMeta)
        try touch(pkg, SmeltBakeArtifacts.grammarTrie)
        try touch(pkg, Qwen3TTSVoice.fileName)
        try touch(pkg, SmeltPackageInterface.fileName)
        let inferred = SmeltBakeManifest.inferExisting(packagePath: pkg)
        XCTAssertEqual(inferred.map(\.kind), [.args, .grammar, .prefix, .voice])
        XCTAssertEqual(
            inferred.first { $0.kind == .grammar }?.perf, [SmeltBakeArtifacts.grammarTrie])
    }

    func testInferExistingGrammarWithoutTrie() throws {
        let pkg = try makeTempPackage()
        try touch(pkg, SmeltBakeArtifacts.grammarMeta)
        let inferred = SmeltBakeManifest.inferExisting(packagePath: pkg)
        XCTAssertEqual(inferred.map(\.kind), [.grammar])
        XCTAssertEqual(inferred.first?.perf, [])
    }

    /// The first marker written on a legacy package is COMPLETE: it backfills
    /// the sidecars already present, not just the component being recorded now.
    func testFirstRecordBackfillsLegacySidecars() throws {
        let pkg = try makeTempPackage()
        try touch(pkg, Qwen3TTSVoice.fileName)
        try touch(pkg, SmeltPackageInterface.fileName)
        try SmeltBakeManifest.record(
            [SmeltBakeManifest.grammar(hasTrie: true)],
            packagePath: pkg)
        let loaded = try XCTUnwrap(try SmeltBakeManifest.load(packagePath: pkg))
        XCTAssertEqual(Set(loaded.sealed.map(\.kind)), [.voice, .args, .grammar])
        XCTAssertEqual(
            loaded.sealed.first { $0.kind == .grammar }?.perf,
            [SmeltBakeArtifacts.grammarTrie])
    }

    // MARK: - CAS-symlink survival (adopt then read)

    func testSymlinkedSidecarIsDetected() throws {
        let pkg = try makeTempPackage()
        // A CAS-adopted artifact is a symlink to a shared blob; a valid symlink
        // must satisfy presence (broken-symlink fail-loud is the U2 enforcement job).
        let blobDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("baketest-blob-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: blobDir, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: blobDir) }
        let blob = blobDir.appendingPathComponent("voice-blob")
        try "{}".write(to: blob, atomically: true, encoding: .utf8)
        try FileManager.default.createSymbolicLink(
            atPath: "\(pkg)/\(Qwen3TTSVoice.fileName)", withDestinationPath: blob.path)
        let inferred = SmeltBakeManifest.inferExisting(packagePath: pkg)
        XCTAssertEqual(inferred.map(\.kind), [.voice])
    }

    // MARK: - Enforcement: presence + closed-world

    func testValidatePresenceDeclaredPresentOK() throws {
        let pkg = try makeTempPackage()
        try touch(pkg, SmeltBakeArtifacts.prefixMeta)
        try touch(pkg, SmeltBakeArtifacts.prefixSnapshot)
        let m = SmeltBakeManifest(sealed: [SmeltBakeManifest.prefix()])
        XCTAssertNoThrow(try m.validatePresence(packagePath: pkg, ignoring: []))
    }

    func testValidatePresenceDeclaredMissingThrows() throws {
        let pkg = try makeTempPackage()
        try touch(pkg, SmeltBakeArtifacts.prefixMeta)  // snapshot missing
        let m = SmeltBakeManifest(sealed: [SmeltBakeManifest.prefix()])
        XCTAssertThrowsError(try m.validatePresence(packagePath: pkg, ignoring: [])) {
            guard case SmeltBakeEnforcementError.declaredArtifactMissing(.prefix, _) = $0 else {
                return XCTFail("expected declaredArtifactMissing, got \($0)")
            }
        }
    }

    func testValidatePresenceClosedWorldThrows() throws {
        let pkg = try makeTempPackage()
        // A grammar sidecar is present but the marker declares only prefix.
        try touch(pkg, SmeltBakeArtifacts.prefixMeta)
        try touch(pkg, SmeltBakeArtifacts.prefixSnapshot)
        try touch(pkg, SmeltBakeArtifacts.grammarMeta)
        let m = SmeltBakeManifest(sealed: [SmeltBakeManifest.prefix()])
        XCTAssertThrowsError(try m.validatePresence(packagePath: pkg, ignoring: [])) {
            guard case SmeltBakeEnforcementError.undeclaredSidecarPresent(_, .grammar) = $0 else {
                return XCTFail("expected undeclaredSidecarPresent, got \($0)")
            }
        }
    }

    func testValidatePresenceIgnoredSkipsBothSides() throws {
        let pkg = try makeTempPackage()
        // prefix declared-but-missing AND grammar present-undeclared — both ignored.
        try touch(pkg, SmeltBakeArtifacts.grammarMeta)
        let m = SmeltBakeManifest(sealed: [SmeltBakeManifest.prefix()])
        XCTAssertNoThrow(
            try m.validatePresence(packagePath: pkg, ignoring: [.prefix, .grammar]))
    }

    func testBrokenSymlinkArtifactFailsLoud() throws {
        let pkg = try makeTempPackage()
        try touch(pkg, SmeltBakeArtifacts.prefixMeta)
        // The snapshot is a symlink to a removed blob (CAS adopt, blob gone).
        try FileManager.default.createSymbolicLink(
            atPath: "\(pkg)/\(SmeltBakeArtifacts.prefixSnapshot)",
            withDestinationPath: "\(pkg)/does-not-exist")
        let m = SmeltBakeManifest(sealed: [SmeltBakeManifest.prefix()])
        XCTAssertThrowsError(try m.validatePresence(packagePath: pkg, ignoring: []))
    }

    // MARK: - TTS enforcement (voice)

    func testVoiceLoadOptOut() throws {
        let pkg = try makeTempPackage()
        try Qwen3TTSVoice(speaker: "ryan").write(packagePath: pkg)
        XCTAssertNotNil(try Qwen3TTSVoice.load(packagePath: pkg, env: [:]))
        XCTAssertNil(try Qwen3TTSVoice.load(
            packagePath: pkg, env: ["SMELT_NO_BAKED_VOICE": "1"]))
    }

    func testEnforceVoiceDeclaredMissingThrows() throws {
        let pkg = try makeTempPackage()
        try SmeltBakeManifest(sealed: [SmeltBakeManifest.voice()]).write(packagePath: pkg)
        XCTAssertThrowsError(
            try SmeltBakeManifest.enforce(packagePath: pkg, ignoring: []))
    }

    func testEnforceVoicePresentOK() throws {
        let pkg = try makeTempPackage()
        try Qwen3TTSVoice(speaker: "ryan").write(packagePath: pkg)
        try SmeltBakeManifest(sealed: [SmeltBakeManifest.voice()]).write(packagePath: pkg)
        XCTAssertNoThrow(
            try SmeltBakeManifest.enforce(packagePath: pkg, ignoring: []))
    }

    func testEnforceVoiceInvalidScheduleThrows() throws {
        let pkg = try makeTempPackage()
        try Qwen3TTSVoice(firstChunkFrames: 2, maxChunkFrames: 1)
            .write(packagePath: pkg)
        try SmeltBakeManifest(sealed: [SmeltBakeManifest.voice()]).write(packagePath: pkg)
        XCTAssertThrowsError(
            try SmeltBakeManifest.enforce(packagePath: pkg, ignoring: [])) {
            XCTAssertTrue(String(describing: $0).contains("maxChunkFrames"))
        }
    }

    /// Opted-out voice is skipped on both sides (declaration + loader).
    func testEnforceVoiceOptOutSkips() throws {
        let pkg = try makeTempPackage()
        try SmeltBakeManifest(sealed: [SmeltBakeManifest.voice()]).write(packagePath: pkg)
        // voice.json missing, but opted out → no throw.
        XCTAssertNoThrow(
            try SmeltBakeManifest.enforce(packagePath: pkg, ignoring: [.voice]))
    }

    /// A declared but corrupt args.json fails loud on every open (serve/linger
    /// don't parse args before GPU construction) — enforce() strict-parses it.
    func testEnforceCorruptArgsThrows() throws {
        let pkg = try makeTempPackage()
        try touch(pkg, SmeltPackageInterface.fileName, "not json")
        try SmeltBakeManifest(sealed: [SmeltBakeManifest.args()]).write(packagePath: pkg)
        XCTAssertThrowsError(
            try SmeltBakeManifest.enforce(packagePath: pkg, ignoring: []))
    }

    func testEnforceValidArgsOK() throws {
        let pkg = try makeTempPackage()
        try touch(pkg, SmeltPackageInterface.fileName,
            "{\"version\":1,\"args\":[{\"flag\":\"x\",\"type\":\"string\",\"target\":\"speaker\"}]}")
        try SmeltBakeManifest(sealed: [SmeltBakeManifest.args()]).write(packagePath: pkg)
        XCTAssertNoThrow(
            try SmeltBakeManifest.enforce(packagePath: pkg, ignoring: []))
    }

    func testEnforceArgsRejectsBuiltinShadowWithManifestContext() throws {
        let pkg = try makeTempPackage()
        try writeDescriptorManifest(
            pkg,
            label: "tts",
            architecture: SmeltRuntimeGraphPolicy.sidecarTextToCodecAudio.rawValue,
            blocks: .qwen3TTSCompiledTrunkNativeFrontEnd
        )
        try touch(pkg, SmeltPackageInterface.fileName,
            "{\"version\":1,\"args\":[{\"flag\":\"speaker\",\"type\":\"string\",\"target\":\"speaker\"}]}")
        try SmeltBakeManifest(sealed: [SmeltBakeManifest.args()]).write(packagePath: pkg)
        XCTAssertThrowsError(
            try SmeltBakeManifest.enforce(packagePath: pkg, ignoring: [])) {
            XCTAssertTrue(String(describing: $0).contains("shadows"))
        }
    }

    // MARK: - Strict loaders (corrupt ⇒ throw, not silent fallback)

    func testGrammarLoadStrictValidThenCorrupt() throws {
        let pkg = try makeTempPackage()
        try SmeltBakedGrammar.write(packagePath: pkg, jsonSchema: "{\"type\":\"object\"}")
        XCTAssertNoThrow(try SmeltBakedGrammar.loadStrict(packagePath: pkg))
        try touch(pkg, SmeltBakedGrammar.fileName, "not json")
        XCTAssertThrowsError(try SmeltBakedGrammar.loadStrict(packagePath: pkg))
    }

    func testGrammarLoadStrictAbsentThrows() throws {
        let pkg = try makeTempPackage()
        XCTAssertThrowsError(try SmeltBakedGrammar.loadStrict(packagePath: pkg))
    }

    func testPrefixLoadStrictCorruptThrows() throws {
        let pkg = try makeTempPackage()
        try touch(pkg, SmeltBakeArtifacts.prefixMeta, "garbage")
        try touch(pkg, SmeltBakeArtifacts.prefixSnapshot, "garbage")
        XCTAssertThrowsError(try SmeltBakedPrefix.loadStrict(packagePath: pkg))
    }

    func testIgnoredFromEnv() {
        // Injectable env — never mutate the real process env (the swift-testing
        // suite runs in parallel and reads SMELT_NO_BAKED_* directly).
        let ignored = SmeltBakeManifest.ignoredFromEnv([
            "SMELT_NO_BAKED_PREFIX": "1",
        ])
        XCTAssertEqual(ignored, [.prefix])
        XCTAssertTrue(SmeltBakeManifest.ignoredFromEnv([:]).isEmpty)
    }

    // MARK: - Single-source filename consistency

    func testFilenameConstantsAreSingleSourced() {
        XCTAssertEqual(SmeltBakeArtifacts.prefixMeta, SmeltBakedPrefix.metaFileName)
        XCTAssertEqual(SmeltBakeArtifacts.prefixSnapshot, SmeltBakedPrefix.snapshotFileName)
        XCTAssertEqual(SmeltBakeArtifacts.grammarMeta, SmeltBakedGrammar.fileName)
        XCTAssertEqual(SmeltBakeArtifacts.grammarTrie, SmeltBakedGrammar.trieFileName)
    }

    // MARK: - Runtime wiring (SmeltModel enforcement; needs the canonical package)

    private static let canonicalPackage =
        "artifacts/qwen35-0.8b-qmm16x128/Qwen_Qwen3.5-0.8B.smeltpkg"

    /// Hardlink-clone the canonical package into a temp dir, stripping any bake
    /// artifacts so it starts unmarked. Returns nil when the package is absent.
    private func cloneCanonical() throws -> String? {
        let fm = FileManager.default
        guard fm.fileExists(atPath: Self.canonicalPackage) else { return nil }
        let clone = fm.temporaryDirectory.appendingPathComponent(
            "bakeenf-\(UUID().uuidString)")
        try fm.createDirectory(at: clone, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: clone) }
        for entry in try fm.contentsOfDirectory(atPath: Self.canonicalPackage)
        where !entry.hasPrefix("baked") {
            let src = "\(Self.canonicalPackage)/\(entry)"
            let dst = clone.appendingPathComponent(entry).path
            var isDir: ObjCBool = false
            fm.fileExists(atPath: src, isDirectory: &isDir)
            if isDir.boolValue {
                try fm.copyItem(atPath: src, toPath: dst)
            } else {
                try fm.linkItem(atPath: src, toPath: dst)
            }
        }

        // The canonical package is an ignored local benchmark artifact and may
        // predate newly tightened manifest bounds. These tests exercise bake
        // enforcement, not release-profile migration, so refresh only the cloned
        // manifest's profile to the current schema before opening the model.
        let manifestURL = clone.appendingPathComponent("manifest.json")
        let manifestData = try Data(contentsOf: manifestURL)
        guard var manifest = try JSONSerialization.jsonObject(with: manifestData)
            as? [String: Any],
            var validation = manifest["validation"] as? [String: Any],
            let gate = validation["performance_gate"] as? String
        else {
            XCTFail("canonical package manifest is missing validation metadata")
            return nil
        }
        let modelName = manifest["modelName"] as? String
        let profile = SmeltPackagePerformanceProfiles.profile(
            for: gate,
            modelName: modelName
        )
        validation["performance_profile"] = try JSONSerialization.jsonObject(
            with: JSONEncoder().encode(profile)
        )
        manifest["validation"] = validation
        try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
            .write(to: manifestURL, options: .atomic)
        return clone.path
    }

    private func gen(_ model: SmeltModel, _ ids: [Int32], _ maxTokens: Int) throws -> [Int32] {
        var n = 0
        return try model.generate(tokenIds: ids) { _ in
            n += 1
            return n < maxTokens
        }.tokens
    }

    /// A correctly marked + baked package loads under enforcement and produces
    /// the same tokens as the unbaked full prefill (no regression).
    func testEnforcedPrefixLoadsAndMatches() throws {
        guard let pkg = try cloneCanonical() else { return }
        let tok = try SmeltTokenizer(path: "\(pkg)/tokenizer.json")
        let prefix = tok.encodeWithSpecials(
            "<|im_start|>system\nBe terse.<|im_end|>\n")
        let full = prefix + tok.encodeWithSpecials(
            "<|im_start|>user\nName one color.<|im_end|>\n<|im_start|>assistant\n")
        let fresh = try SmeltModel(package: pkg)
        let baseline = try gen(fresh, full, 8)
        try fresh.bakePromptPrefix(tokenIds: prefix)
        try SmeltBakeManifest(sealed: [SmeltBakeManifest.prefix()]).write(packagePath: pkg)
        let baked = try SmeltModel(package: pkg)
        XCTAssertEqual(baked.bakedPrefixTokenIds, prefix)
        XCTAssertEqual(try gen(baked, full, 8), baseline)
    }

    /// A marker declaring a prefix whose snapshot is missing must fail loud at
    /// model open, not silently fall back.
    func testDeclaredMissingPrefixFailsLoud() throws {
        guard let pkg = try cloneCanonical() else { return }
        let tok = try SmeltTokenizer(path: "\(pkg)/tokenizer.json")
        let prefix = tok.encodeWithSpecials(
            "<|im_start|>system\nBe terse.<|im_end|>\n")
        let fresh = try SmeltModel(package: pkg)
        try fresh.bakePromptPrefix(tokenIds: prefix)
        try SmeltBakeManifest(sealed: [SmeltBakeManifest.prefix()]).write(packagePath: pkg)
        try FileManager.default.removeItem(
            atPath: "\(pkg)/\(SmeltBakedPrefix.snapshotFileName)")
        XCTAssertThrowsError(try SmeltModel(package: pkg))
    }

    /// A recognized sidecar present but not declared (partial/failed bake) must
    /// fail loud (closed-world).
    func testUndeclaredSidecarFailsLoud() throws {
        guard let pkg = try cloneCanonical() else { return }
        let tok = try SmeltTokenizer(path: "\(pkg)/tokenizer.json")
        let prefix = tok.encodeWithSpecials(
            "<|im_start|>system\nBe terse.<|im_end|>\n")
        let fresh = try SmeltModel(package: pkg)
        try fresh.bakePromptPrefix(tokenIds: prefix)
        try SmeltBakeManifest(sealed: [SmeltBakeManifest.prefix()]).write(packagePath: pkg)
        // Marker declares prefix only; a stray grammar sidecar is present.
        try touch(pkg, SmeltBakedGrammar.fileName,
            "{\"version\":1,\"json_schema\":\"{}\"}")
        XCTAssertThrowsError(try SmeltModel(package: pkg))
    }

    /// Enforcement lives at the universal open point (SmeltRuntime.init), so a
    /// decode-only / serve path that opens the runtime directly — without ever
    /// constructing SmeltModel — still fails loud.
    func testDirectRuntimeOpenEnforces() throws {
        guard let pkg = try cloneCanonical() else { return }
        try SmeltBakeManifest(sealed: [SmeltBakeManifest.grammar(hasTrie: false)])
            .write(packagePath: pkg)
        XCTAssertThrowsError(try SmeltRuntime(packagePath: pkg))
    }

    /// A stray grammar trie (grammar not declared) is a recognized perf sidecar
    /// and trips the closed-world check.
    func testStrayTrieFailsLoud() throws {
        let pkg = try makeTempPackage()
        try touch(pkg, SmeltBakeArtifacts.prefixMeta)
        try touch(pkg, SmeltBakeArtifacts.prefixSnapshot)
        try touch(pkg, SmeltBakeArtifacts.grammarTrie)
        let m = SmeltBakeManifest(sealed: [SmeltBakeManifest.prefix()])
        XCTAssertThrowsError(try m.validatePresence(packagePath: pkg, ignoring: [])) {
            guard case SmeltBakeEnforcementError.undeclaredSidecarPresent(_, .grammar) = $0
            else { return XCTFail("expected undeclaredSidecarPresent(.grammar), got \($0)") }
        }
    }

    /// File-level closed-world: a grammar marker that omits the trie from `perf`
    /// while the trie is on disk is a present-but-undeclared perf artifact.
    func testGrammarDeclaredWithoutTrieButTriePresentFailsLoud() throws {
        let pkg = try makeTempPackage()
        try touch(pkg, SmeltBakeArtifacts.grammarMeta)
        try touch(pkg, SmeltBakeArtifacts.grammarTrie)
        let m = SmeltBakeManifest(sealed: [SmeltBakeManifest.grammar(hasTrie: false)])
        XCTAssertThrowsError(try m.validatePresence(packagePath: pkg, ignoring: []))
    }

    func testGrammarDeclaredWithTriePresentOK() throws {
        let pkg = try makeTempPackage()
        try touch(pkg, SmeltBakeArtifacts.grammarMeta)
        try touch(pkg, SmeltBakeArtifacts.grammarTrie)
        let m = SmeltBakeManifest(sealed: [SmeltBakeManifest.grammar(hasTrie: true)])
        XCTAssertNoThrow(try m.validatePresence(packagePath: pkg, ignoring: []))
    }
}
