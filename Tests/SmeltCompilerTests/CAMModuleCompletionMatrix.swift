import CryptoKit
import Foundation
import SmeltCompiler
import SmeltSchema

enum CAMModuleCompletionMatrix {
    static let selectorDeletionScannerVersion = 10
    static let runtimeAssemblyFeatureContractSchema = "smelt.module.runtime_assembly_feature_contract.v1"

    enum CompletionLevel: Int, CaseIterable, Comparable, CustomStringConvertible {
        case authored = 0
        case capabilityResolved
        case checkedPackageProjected
        case buildCommandCovered
        case releaseGated
        case runtimeContractCovered
        case capabilityRouted
        case runtimeAssemblyCovered
        case selectorRootsDeleted

        static func < (lhs: CompletionLevel, rhs: CompletionLevel) -> Bool {
            lhs.rawValue < rhs.rawValue
        }

        var description: String {
            switch self {
            case .authored:
                return "authored"
            case .capabilityResolved:
                return "capability-resolved"
            case .checkedPackageProjected:
                return "checked-package-projected"
            case .buildCommandCovered:
                return "build-command-covered"
            case .releaseGated:
                return "release-gated"
            case .runtimeContractCovered:
                return "runtime-contract-covered"
            case .capabilityRouted:
                return "capability-routed"
            case .runtimeAssemblyCovered:
                return "runtime-assembly-covered"
            case .selectorRootsDeleted:
                return "selector-roots-deleted"
            }
        }
    }

    struct CompletionTarget: Sendable {
        let id: String
        let fixtures: [String]
    }

    enum TransitionReleaseSourceAvailability: String, Sendable {
        case allSourceRefsPresent = "all-source-refs-present"
        case never
    }

    struct TransitionReleaseBlocker: Sendable, Equatable {
        let code: String
        let message: String
        let targetSurfaceIDs: [String]
        let targetRequiredGateIDs: [String]
        let rowSurfaceIDs: [String]
    }

    struct TransitionReleaseSourcePolicy: Sendable, Equatable {
        let schema: Int
        let transitionOnly: Bool
        let availability: TransitionReleaseSourceAvailability
        let sourceRefs: [String]
        let sourceRefsBySurface: [String: [String]]
        let availableBlocker: TransitionReleaseBlocker?
        let missingSourceBlocker: TransitionReleaseBlocker?
        let staticBlockers: [TransitionReleaseBlocker]
    }

    struct CapabilityExpectation {
        let request: SmeltCAMCapabilityRequest
        let exportID: String
        let flowID: String
        let selectedInputs: [String]
        let selectedOutputs: [String]
        var selectedInputNames: [String] = []
        var selectedOutputNames: [String] = []
        let gateIDs: [String]
        let gateRequirements: [String]
        let gatePredicates: [String]
        var surfaceID: String? = nil
    }

    struct PackageProjectionExpectation: Sendable {
        let id: String
        let version: Int
        let projectedPackageSpecSHA256: String
        let packageFiles: [String]
        var releaseBuildEvidenceSHA256: String? = nil
        var releasePackagePayloadSHA256: String? = nil
    }

    struct RuntimeContractExpectation {
        let surfaceID: String
        let evidenceID: String
        let request: SmeltCAMCapabilityRequest
        let exportID: String
        let flowID: String
        let gateIDs: [String]
        let bodySHA256: String
        let provenanceSHA256: String
    }

    struct CommandSurfaceObligationExpectation: Sendable, Equatable {
        let surfaceID: String
        let command: String
        let subcommand: String
        let runtimeEventSource: String
        let deliveryMode: String
        let counted: Bool
        let requiresRuntimeContract: Bool
        let requiresRuntimeAssembly: Bool
        let bridgeAbsentRequired: Bool
    }

    struct EvidenceObligationExpectation: Sendable, Equatable {
        let obligationKey: String
        let surfaceID: String
        let commandSurfaceObligationKey: String
        let command: String
        let subcommand: String
        let runtimeEventSource: String
        let deliveryMode: String
        let counted: Bool
        let role: String
        let exportID: String?
        let flowID: String?
        let gateIDs: [String]
        let selectedInputNames: [String]
        let selectedInputs: [String]
        let selectedOutputNames: [String]
        let selectedOutputs: [String]
        let requiresReleaseEvidence: Bool
        let requiresRuntimeContract: Bool
        let requiresRuntimeAssembly: Bool
        let adapterBoundary: AdapterBoundary
        var runtimeAssemblyFeatureContractSchema: String? = nil
        var configuredGraphFeatureSet: [String] = []
        var requiredFeatureSet: [String] = []
        var consumedFeatureSet: [String] = []
        var unsupportedFeatureSet: [String] = []
        var requiredObligationIDs: [String] = []
        var consumedObligationIDs: [String] = []
        var unsupportedObligationIDs: [String] = []
    }

    enum SurfaceKind: String, Sendable {
        case build
        case command
        case gate
        case correctness
        case optionalInputCoverage
        case releaseVerification
    }

    enum AdapterBoundary: String, Sendable, Equatable {
        case none
        case checkedProjectionProfile = "projection-check"
        case manifestPolicyBridge = "policy-adapter"
        case commandAdapter = "command-adapter"
        case releaseBucketAdapter = "release-evidence-adapter"
        case targetVerifierBucket = "verification-adapter"
    }

    enum ProductionSelectorScope: String, CaseIterable, Sendable {
        case build
        case release
        case dispatch
        case command
        case assembly
        case runtime
    }

    struct SurfaceExpectation: Sendable, Equatable {
        let surfaceID: String
        let kind: SurfaceKind
        let request: SmeltCAMCapabilityRequest?
        let exportID: String?
        let flowID: String?
        let selectedInputs: [String]
        let selectedOutputs: [String]
        let selectedInputNames: [String]
        let selectedOutputNames: [String]
        let gateIDs: [String]
        let requiresRuntimeContract: Bool
        let requiresReleaseEvidence: Bool
        let requiresRuntimeAssembly: Bool
        let adapterBoundary: AdapterBoundary
    }

    struct ReleaseGateExpectation: Sendable, Equatable {
        let surfaceID: String
        let evidenceID: String
        let request: SmeltCAMCapabilityRequest?
        let exportID: String
        let flowID: String
        let selectedCapability: String
        let gateID: String
        let gateTraceLabels: [String]
        let metric: String
        let metricPath: String
        let comparator: String
        let bound: String
        let unit: String?
        let fromEventID: String
        let toEventID: String
        let processMode: String
        let processStartTimestamp: String
        let cacheState: String
        let camSemanticSHA256: String
        let exportABISHA256: String
        let camDescriptorSHA256: String
        let descriptorGraphSignatureSHA256: String
        let packageProjectionID: String
        let packageProjectionVersion: Int
        let projectedPackageSpecSHA256: String
        let packageFiles: [String]
        let buildEvidenceSHA256: String
        let packagePayloadSHA256: String
        let commandStamp: String
        let measurementSHA256: String
        let provenanceSHA256: String
    }

    struct RuntimeAssemblyExpectation: Sendable, Equatable {
        let surfaceID: String
        let evidenceID: String
        let request: SmeltCAMCapabilityRequest
        let exportID: String
        let flowID: String
        let assemblyPlanKey: String
        let executionBackend: String
        let artifactRoles: [String]
        let gateIDs: [String]
        let requiredFeatureSet: [String]
        let consumedFeatureSet: [String]
        let unsupportedFeatureSet: [String]
        let requiredObligationIDs: [String]
        let consumedObligationIDs: [String]
        let unsupportedObligationIDs: [String]
        let selectorDeletionLintID: String
        let manifestPolicyBridgeAbsent: Bool
    }

    struct SelectorDeletionScanExpectation: Sendable, Equatable {
        let lintID: String
        let scannerVersion: Int
        let command: [String]
        let sourceRootSHA256: String
        let allowlistSHA256: String
        let provenanceSHA256: String
        let scopes: [ProductionSelectorScope]
        let requiredPathClasses: [String]
        let scannedPathClasses: [String]
        let scannedFileKinds: [SelectorDeletionScannedFileKind]
        let scannedPaths: [String]
        let prohibitedHits: [String]
    }

    struct SelectorDeletionScannedFileKind: Sendable, Equatable {
        let kind: String
        let count: Int
    }

    struct ClosedSurfaceExpectation: Sendable, Equatable {
        let surfaceID: String
        let surface: String
        let reason: String
        let requiredGateIDs: [String]
    }

    struct PublicOptionalInputExpectation: Sendable, Equatable {
        let surfaceID: String
        let exportID: String
        let flowID: String
        let portName: String
        let portShape: String
        let defaultSemantics: String
        let acceptedValueSemantics: String
        let closedSurfaceEvidenceToken: String
        let negativeSemantics: String
        let gateIDs: [String]
    }

    private struct RuntimeContractHashes {
        let bodySHA256: String
        let provenanceSHA256: String
    }

    private static let textPackageFiles = [
        "SmeltGenerated.swift",
        "dispatches.bin",
        "manifest.json",
        "model.metallib",
        "module.json",
        "prefill_dispatches.bin",
        "tokenizer.bin",
        "tokenizer.json",
        "weights.bin",
    ]

    static let releaseGateContractSchema = SmeltCAMReleaseGateContract.currentSchema

    struct Module {
        let fixture: String
        let capabilityTrack: String
        let level: CompletionLevel
        let semanticSHA256: String
        let exportABISHA256: String
        let descriptorGraphSignatureSHA256: String
        let requiredCapabilities: [String]
        let requiredGates: [String]
        let requiredReleaseGates: [String]
        let surfaceExpectations: [SurfaceExpectation]
        let capabilityExpectations: [CapabilityExpectation]
        let packageProjection: PackageProjectionExpectation?
        var releaseGateEvidence: [ReleaseGateExpectation] = []
        var runtimeContractEvidence: [RuntimeContractExpectation] = []
        var runtimeAssemblyEvidence: [RuntimeAssemblyExpectation] = []
        var selectorDeletionScans: [SelectorDeletionScanExpectation] = []
        var closedSurfaces: [ClosedSurfaceExpectation] = []
        var publicOptionalInputs: [PublicOptionalInputExpectation] = []
        var consumedAdmissionObligationIDs: [String] = []

        var targetID: String {
            guard fixture.hasSuffix(".module.json") else { return fixture }
            return String(fixture.dropLast(".module.json".count))
        }

        var isReleaseGatedOrHigher: Bool {
            level >= .releaseGated
        }
    }

    static var completionTargets: [CompletionTarget] {
        modules.map { module in
            CompletionTarget(id: module.targetID, fixtures: [module.fixture])
        }
    }

    static let modules: [Module] = [
        .init(
            fixture: "qwen35_text.module.json",
            capabilityTrack: "text-generation",
            level: .selectorRootsDeleted,
            semanticSHA256: "e23f15dda6ad25af95a564c4be8b64356b9f1024c8a7d9e6aabe3c141e1ed859",
            exportABISHA256: "8052ea9d754e9633900dbaed98b03cc988552072c629b1f456520fcf6025bfc2",
            descriptorGraphSignatureSHA256: "77845c4dfb5085685611ef687da013a5d802c3154f529fee524f375cc62aa421",
            requiredCapabilities: ["run.generate"],
            requiredGates: ["startup", "prefill", "decode", "inventory"],
            requiredReleaseGates: ["startup", "prefill", "decode"],
            surfaceExpectations: textSurfaceExpectations(
                includeBake: false,
                releaseGateIDs: ["startup", "prefill", "decode"],
                buildAdapterBoundary: AdapterBoundary.none,
                commandAdapterBoundary: AdapterBoundary.none,
                lingerAdapterBoundary: AdapterBoundary.none,
                inventoryGateAdapterBoundary: AdapterBoundary.none,
                releaseGateAdapterBoundary: AdapterBoundary.none,
                releaseVerificationAdapterBoundary: AdapterBoundary.none
            ),
            capabilityExpectations: textGenerationExpectations(
                elapsedMs: "115",
                bakePromptPrefix: false,
                includeLinger: true
            ),
            packageProjection: .init(
                id: "text-to-text-transformer-prefill-decode-affine-u4-g64",
                version: 1,
                projectedPackageSpecSHA256: "1e023320c723d12534cb1a0ad398bac237bf0d2fa21fb4ae34849e42109a872c",
                packageFiles: [
                    "SmeltGenerated.swift",
                    "dispatches.bin",
                    "manifest.json",
                    "model.metallib",
                    "module.json",
                    "prefill_dispatches.bin",
                    "tokenizer.bin",
                    "tokenizer.json",
                    "weights.bin",
                ],
                releaseBuildEvidenceSHA256: "",
                releasePackagePayloadSHA256: ""
            ),
            releaseGateEvidence: textReleaseGateEvidence(
                packageFiles: [
                    "SmeltGenerated.swift",
                    "dispatches.bin",
                    "manifest.json",
                    "model.metallib",
                    "module.json",
                    "prefill_dispatches.bin",
                    "tokenizer.bin",
                    "tokenizer.json",
                    "weights.bin",
                ],
                camSemanticSHA256: "e23f15dda6ad25af95a564c4be8b64356b9f1024c8a7d9e6aabe3c141e1ed859",
                exportABISHA256: "8052ea9d754e9633900dbaed98b03cc988552072c629b1f456520fcf6025bfc2",
                camDescriptorSHA256: "0d409b03c30f5ac0d4f4ade8b82c37f28f159c81a64cc63c10c3f43052e21f64",
                descriptorGraphSignatureSHA256: "77845c4dfb5085685611ef687da013a5d802c3154f529fee524f375cc62aa421",
                packageProjectionID: "text-to-text-transformer-prefill-decode-affine-u4-g64",
                projectedPackageSpecSHA256: "1e023320c723d12534cb1a0ad398bac237bf0d2fa21fb4ae34849e42109a872c",
                buildEvidenceSHA256: "",
                packagePayloadSHA256: "",
                elapsedBound: "115",
                includePrefillDecode: true
            ),
            runtimeContractEvidence: textRuntimeContractEvidence(
                exportGateIDs: ["startup"],
                includeBake: false,
                includeLinger: true,
                run: .init(
                    bodySHA256: "232cab44f69a7fd7a64d620a0bdd82d7d09eb846fb15fbae67a1d760835315e2",
                    provenanceSHA256: "263fa6176090e032727cb3b2e2f98c14c5129402e6178d3325dfc59a59b494ad"
                )
            ),
            runtimeAssemblyEvidence: textRuntimeAssemblyEvidence(
                fixture: "qwen35_text.module.json",
                includeBake: false,
                includeLinger: true
            ),
            selectorDeletionScans: productionSelectorDeletionScans(
                fixture: "qwen35_text.module.json"
            )
        ),
        .init(
            fixture: "qwen35_fast.module.json",
            capabilityTrack: "text-generation",
            level: .selectorRootsDeleted,
            semanticSHA256: "38701983fb3cd91109681c11f033e377c535256fa9e9aac3ec5129e37d40a301",
            exportABISHA256: "f3f7a7e4d99536a305b5908fe14f89734dda6bb6aea86ab9b90b59859f733e4f",
            descriptorGraphSignatureSHA256: "77845c4dfb5085685611ef687da013a5d802c3154f529fee524f375cc62aa421",
            requiredCapabilities: ["run.generate"],
            requiredGates: ["startup", "prefill", "decode", "inventory"],
            requiredReleaseGates: ["startup", "prefill", "decode"],
            surfaceExpectations: textSurfaceExpectations(
                includeBake: false,
                releaseGateIDs: ["startup", "prefill", "decode"],
                buildAdapterBoundary: .none,
                commandAdapterBoundary: .none,
                lingerAdapterBoundary: AdapterBoundary.none,
                inventoryGateAdapterBoundary: AdapterBoundary.none,
                releaseGateAdapterBoundary: .none,
                releaseVerificationAdapterBoundary: .none
            ),
            capabilityExpectations: textGenerationExpectations(
                elapsedMs: "100",
                bakePromptPrefix: false,
                includeLinger: true
            ),
            packageProjection: .init(
                id: "text-to-text-transformer-prefill-decode-affine-u4-g64",
                version: 1,
                projectedPackageSpecSHA256: "5a27a610102ad6c01180db91a340644bf3dba27e8663e4d95137a423aaba1c44",
                packageFiles: [
                    "SmeltGenerated.swift",
                    "dispatches.bin",
                    "manifest.json",
                    "model.metallib",
                    "module.json",
                    "prefill_dispatches.bin",
                    "tokenizer.bin",
                    "tokenizer.json",
                    "weights.bin",
                ],
                releaseBuildEvidenceSHA256: "",
                releasePackagePayloadSHA256: ""
            ),
            releaseGateEvidence: textReleaseGateEvidence(
                packageFiles: [
                    "SmeltGenerated.swift",
                    "dispatches.bin",
                    "manifest.json",
                    "model.metallib",
                    "module.json",
                    "prefill_dispatches.bin",
                    "tokenizer.bin",
                    "tokenizer.json",
                    "weights.bin",
                ],
                camSemanticSHA256: "38701983fb3cd91109681c11f033e377c535256fa9e9aac3ec5129e37d40a301",
                exportABISHA256: "f3f7a7e4d99536a305b5908fe14f89734dda6bb6aea86ab9b90b59859f733e4f",
                camDescriptorSHA256: "dd8a286215db439e39c16353d80f238a85dac312a5695faf9a4079008e74d5b7",
                descriptorGraphSignatureSHA256: "77845c4dfb5085685611ef687da013a5d802c3154f529fee524f375cc62aa421",
                packageProjectionID: "text-to-text-transformer-prefill-decode-affine-u4-g64",
                projectedPackageSpecSHA256: "5a27a610102ad6c01180db91a340644bf3dba27e8663e4d95137a423aaba1c44",
                buildEvidenceSHA256: "",
                packagePayloadSHA256: "",
                elapsedBound: "100",
                includePrefillDecode: true
            ),
            runtimeContractEvidence: textRuntimeContractEvidence(
                exportGateIDs: ["startup"],
                includeBake: false,
                includeLinger: true,
                run: .init(
                    bodySHA256: "8f7d63b370b60d1a4d6decbc87da257137670c16fc8e3549d68a19957496f4f1",
                    provenanceSHA256: "fd52e94a3b5c71f00736a7bb176e69c55956d487bd2f978e7a4d4e89634061d8"
                )
            ),
            runtimeAssemblyEvidence: textRuntimeAssemblyEvidence(
                fixture: "qwen35_fast.module.json",
                includeLinger: true
            ),
            selectorDeletionScans: productionSelectorDeletionScans(
                fixture: "qwen35_fast.module.json"
            ),
            closedSurfaces: []
        ),
        .init(
            fixture: "qwen3_tts.module.json",
            capabilityTrack: "streaming-text-to-24khz-audio",
            level: .selectorRootsDeleted,
            semanticSHA256: "b5adbe257f202d54c025568111448b11ca3abd0a50e479839602c3c08b56ee31",
            exportABISHA256: "691518abe0d2984d67f8c4eaaa1173c44d7163eba13a46d67201e5b05e86c8dc",
            descriptorGraphSignatureSHA256: "0d85708b0dc274ac0138e1df7d3d1bc2269392191671245d8028e2ef6046d41d",
            requiredCapabilities: ["run.stream", "run.synthesize"],
            requiredGates: ["startup", "audio_contract", "streaming_parity"],
            requiredReleaseGates: ["startup", "audio_contract", "streaming_parity"],
            surfaceExpectations: textToAudio24KSurfaceExpectations(
                commandAdapterBoundary: .none,
                traceRequiresRuntimeContract: true,
                traceAdapterBoundary: .none,
                optionalInputAdapterBoundary: .none,
                buildAdapterBoundary: .none,
                releaseCommandAdapterBoundary: .none,
                releaseGateAdapterBoundary: .none,
                releaseVerificationAdapterBoundary: .none
            ),
            capabilityExpectations: textToAudio24KExpectations(),
            packageProjection: .init(
                id: "streaming-text-to-24khz-audio-derived-manifest-affine-u4-g128-sidecars",
                version: 1,
                // Release-build pin from the package plan's relative TTS artifact root.
                projectedPackageSpecSHA256: "bd2d7888fe7a5d5eda660f9faef3fa87d7201919b2252ad9251b617005e54a71",
                packageFiles: [
                    "config.json",
                    "manifest.json",
                    "merges.txt",
                    "model.metallib",
                    "module.json",
                    "tokenizer_config.json",
                    "trunk",
                    "trunk-mtp",
                    "vocab.json",
                    "weights.bin",
                ],
                releaseBuildEvidenceSHA256: "",
                releasePackagePayloadSHA256: ""
            ),
            releaseGateEvidence: audio24KReleaseGateEvidence(
                packageFiles: [
                    "config.json",
                    "manifest.json",
                    "merges.txt",
                    "model.metallib",
                    "module.json",
                    "tokenizer_config.json",
                    "trunk",
                    "trunk-mtp",
                    "vocab.json",
                    "weights.bin",
                ],
                camSemanticSHA256: "b5adbe257f202d54c025568111448b11ca3abd0a50e479839602c3c08b56ee31",
                exportABISHA256: "691518abe0d2984d67f8c4eaaa1173c44d7163eba13a46d67201e5b05e86c8dc",
                camDescriptorSHA256: "9d0d39bdbdfeea04cf8b12cfb70e6fa7f60454d9d6b1ad7d4e735107f240730d",
                descriptorGraphSignatureSHA256: "0d85708b0dc274ac0138e1df7d3d1bc2269392191671245d8028e2ef6046d41d",
                packageProjectionID: "streaming-text-to-24khz-audio-derived-manifest-affine-u4-g128-sidecars",
                projectedPackageSpecSHA256: "bd2d7888fe7a5d5eda660f9faef3fa87d7201919b2252ad9251b617005e54a71",
                buildEvidenceSHA256: "",
                packagePayloadSHA256: ""
            ),
            runtimeContractEvidence: audioRuntimeContractEvidence(
                exportID: "synth",
                flowID: "synth",
                includeServe: true,
                includeBake: false,
                includeTrace: true,
                includeLinger: true,
                run: .init(
                    bodySHA256: "7f46b0c185e2d3d14b70cf7cb92f2dba544192d1fcf1c9ae8eccfdeb238bf64b",
                    provenanceSHA256: "2cf79eb777b3d32796c25d1f07a075e27745ab2c5a3dea8ec80667351f2d4483"
                )
            ),
            runtimeAssemblyEvidence: audio24KRuntimeAssemblyEvidence(
                fixture: "qwen3_tts.module.json",
                exportID: "synth",
                flowID: "synth",
                includeServe: true,
                includeTrace: true,
                includeLinger: true
            ),
            selectorDeletionScans: productionSelectorDeletionScans(
                fixture: "qwen3_tts.module.json"
            ),
            closedSurfaces: textToAudio24KClosedSurfaces(
                includeRuntimeCommandBlockers: false,
                includeTraceCommandBlocker: false,
                includeReleaseEvidenceBlockers: false,
                includeOptionalInputBlocker: false
            ),
            publicOptionalInputs: [
                .init(
                    surfaceID: "optional.speaker",
                    exportID: "synth",
                    flowID: "synth",
                    portName: "speaker",
                    portShape: "voice-id",
                    defaultSemantics: "absent speaker uses package default voice defaults",
                    acceptedValueSemantics: "named speaker resolves through CAM voice-default spk_id contract",
                    closedSurfaceEvidenceToken: "optional speaker selection",
                    negativeSemantics: "unsupported speaker id fails closed before runtime construction",
                    gateIDs: ["startup"]
                ),
            ]
        ),
    ]

    static var fixtureNames: [String] {
        modules.map(\.fixture)
    }

    static var checkedProjectedFixtureNames: [String] {
        modules
            .filter { $0.level >= .checkedPackageProjected }
            .map(\.fixture)
    }

    static var buildCommandCoveredFixtureNames: [String] {
        modules
            .filter { $0.level >= .buildCommandCovered }
            .map(\.fixture)
    }

    static var releaseGatedFixtureNames: [String] {
        modules
            .filter { $0.level >= .releaseGated }
            .map(\.fixture)
    }

    static var runtimeContractCoveredFixtureNames: [String] {
        modules
            .filter { $0.level >= .runtimeContractCovered }
            .map(\.fixture)
    }

    static var unprojectedFixtureNames: [String] {
        modules
            .filter { $0.level < .checkedPackageProjected }
            .map(\.fixture)
    }

    static var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    static func module(fixture: String) throws -> Module {
        guard let module = modules.first(where: { $0.fixture == fixture }) else {
            throw MatrixError.unknownFixture(fixture)
        }
        return module
    }

    static func commandSurfaceObligations(
        for module: Module
    ) -> [CommandSurfaceObligationExpectation] {
        module.surfaceExpectations.flatMap(commandSurfaceObligations(for:))
    }

    static func runtimeContractObligations(
        for module: Module
    ) throws -> [EvidenceObligationExpectation] {
        try evidenceObligations(for: module, includeRuntimeAssemblyFeatureSets: false) {
            $0.requiresRuntimeContract
        }
    }

    static func runtimeAssemblyObligations(
        for module: Module
    ) throws -> [EvidenceObligationExpectation] {
        try evidenceObligations(for: module, includeRuntimeAssemblyFeatureSets: true) {
            $0.requiresRuntimeAssembly
        }
    }

    static func configuredGraphFeatures(for module: Module) throws -> [String] {
        let obligations = try runtimeAssemblyObligations(for: module)
        return Array(Set(obligations.flatMap(\.configuredGraphFeatureSet))).sorted()
    }

    static func transitionReleaseSourcePolicy(
        for module: Module
    ) -> TransitionReleaseSourcePolicy {
        let releaseSurfaceRefs = releaseSurfaceRefs(for: module)
        if module.isReleaseGatedOrHigher {
            return .init(
                schema: 1,
                transitionOnly: false,
                availability: .never,
                sourceRefs: [],
                sourceRefsBySurface: [:],
                availableBlocker: nil,
                missingSourceBlocker: nil,
                staticBlockers: []
            )
        }
        let sourceRefs: [String]
        let sourceRefsBySurface: [String: [String]]
        let missingSourceBlocker: TransitionReleaseBlocker?
        let staticBlockers: [TransitionReleaseBlocker]
        let availability: TransitionReleaseSourceAvailability
        switch module.fixture {
        case "qwen35_text.module.json", "qwen35_fast.module.json":
            availability = .allSourceRefsPresent
            sourceRefs = [
                "raw_inputs.transition_startup",
                "raw_inputs.transition_benchmark",
                "raw_inputs.transition_verify",
                "raw_inputs.transition_package_manifests",
            ]
            sourceRefsBySurface = [:]
            missingSourceBlocker = nil
            staticBlockers = []
        case "qwen3_tts.module.json":
            availability = .allSourceRefsPresent
            sourceRefsBySurface = [
                "\(module.fixture):correctness.stream-parity": [
                    "transition.voice.correctness.stream-parity"
                ],
                "\(module.fixture):gate.audio-contract": ["transition.voice.gate.audio-contract"],
                "\(module.fixture):gate.startup-audio": ["transition.voice.gate.startup-audio"],
                "\(module.fixture):release.verify": ["transition.voice.release.verify"],
            ]
            sourceRefs = Array(Set(sourceRefsBySurface.values.flatMap { $0 })).sorted()
            missingSourceBlocker = .init(
                code: "missing-transition-release-source",
                message: "transition release source evidence was not provided",
                targetSurfaceIDs: releaseSurfaceRefs,
                targetRequiredGateIDs: module.requiredReleaseGates,
                rowSurfaceIDs: releaseSurfaceRefs
            )
            staticBlockers = []
        default:
            availability = .never
            sourceRefs = []
            sourceRefsBySurface = [:]
            missingSourceBlocker = nil
            staticBlockers = [
                .init(
                    code: "missing-transition-release-source-policy",
                    message: "module has no transition release source policy",
                    targetSurfaceIDs: releaseSurfaceRefs,
                    targetRequiredGateIDs: module.requiredReleaseGates,
                    rowSurfaceIDs: releaseSurfaceRefs
                ),
            ]
        }
        let availableBlocker: TransitionReleaseBlocker? = sourceRefs.isEmpty ? nil : .init(
            code: "transition-release-source-evidence",
            message: "release evidence is still produced by transition source inputs",
            targetSurfaceIDs: releaseSurfaceRefs,
            targetRequiredGateIDs: module.requiredReleaseGates,
            rowSurfaceIDs: releaseSurfaceRefs
        )
        return .init(
            schema: 1,
            transitionOnly: true,
            availability: availability,
            sourceRefs: sourceRefs,
            sourceRefsBySurface: sourceRefsBySurface,
            availableBlocker: availableBlocker,
            missingSourceBlocker: missingSourceBlocker,
            staticBlockers: staticBlockers
        )
    }

    private static func productionSelectorDeletionScans(
        fixture: String,
        lintID: String = "cam-block-graph-clean"
    ) -> [SelectorDeletionScanExpectation] {
        // The selector-deletion scanner is private maintainer machinery
        // (export-ignore in .gitattributes); the public tree ships this matrix
        // without it, and the public consumers of `modules` never read
        // `selectorDeletionScans`. Returning [] keeps the shared static
        // constructible in the public tree. This is NOT a silent skip: the
        // private suite that asserts scan expectations
        // (CAMModuleCompletionMatrixTests.selectorRootDeletedMatrixScansReproduceFromScanner)
        // requires a non-empty scan list per module, so a missing scanner in
        // the private tree fails loudly there.
        let scanner = repoRoot
            .appendingPathComponent("tools/check-module-selector-deletion")
        guard FileManager.default.fileExists(atPath: scanner.path) else {
            return []
        }
        return [runProductionSelectorDeletionScan(fixture: fixture, lintID: lintID)]
    }

    private static func runProductionSelectorDeletionScan(
        fixture: String,
        lintID: String
    ) -> SelectorDeletionScanExpectation {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = [
            "python3",
            "tools/check-module-selector-deletion",
            "--fixture",
            fixture,
            "--lint-id",
            lintID,
            "--format",
            "row-json",
        ]
        process.currentDirectoryURL = repoRoot

        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr
        do {
            try process.run()
        } catch {
            fatalError("failed to run selector deletion scanner: \(error)")
        }
        process.waitUntilExit()

        let stdoutData = stdout.fileHandleForReading.readDataToEndOfFile()
        let stderrText = String(
            decoding: stderr.fileHandleForReading.readDataToEndOfFile(),
            as: UTF8.self
        )
        guard process.terminationStatus == 0 else {
            fatalError("selector deletion scanner failed: \(stderrText)")
        }
        do {
            guard let row = try JSONSerialization.jsonObject(with: stdoutData) as? [String: Any] else {
                fatalError("selector deletion scanner emitted non-object JSON")
            }
            return try selectorDeletionScanExpectation(from: row, fixture: fixture, lintID: lintID)
        } catch {
            fatalError("selector deletion scanner emitted invalid evidence: \(error)")
        }
    }

    private static func selectorDeletionScanExpectation(
        from row: [String: Any],
        fixture: String,
        lintID: String
    ) throws -> SelectorDeletionScanExpectation {
        func require<T>(_ key: String, as type: T.Type = T.self) throws -> T {
            guard let value = row[key] as? T else {
                throw MatrixError.invalidSelectorDeletionScan("\(fixture): \(key)")
            }
            return value
        }

        let fixtureValue: String = try require("fixture")
        guard fixtureValue == fixture else {
            throw MatrixError.invalidSelectorDeletionScan("\(fixture): fixture mismatch")
        }
        let lintIDs: [String] = try require("lint_ids")
        guard lintIDs == [lintID] else {
            throw MatrixError.invalidSelectorDeletionScan("\(fixture): lint ids mismatch")
        }
        let scopes: [ProductionSelectorScope] = try require("production_scopes", as: [String].self)
            .map { rawValue in
                guard let scope = ProductionSelectorScope(rawValue: rawValue) else {
                    throw MatrixError.invalidSelectorDeletionScan("\(fixture): unknown scope \(rawValue)")
                }
                return scope
            }
        let fileKinds: [SelectorDeletionScannedFileKind] =
            try require("scanned_file_kinds", as: [[String: Any]].self).map { item in
                guard let kind = item["kind"] as? String,
                      let count = item["count"] as? Int else {
                    throw MatrixError.invalidSelectorDeletionScan("\(fixture): invalid file kind")
                }
                return .init(kind: kind, count: count)
            }
        return .init(
            lintID: lintID,
            scannerVersion: try require("scanner_version"),
            command: try require("command"),
            sourceRootSHA256: try require("source_root_sha256"),
            allowlistSHA256: try require("allowlist_sha256"),
            provenanceSHA256: try require("provenance_sha256"),
            scopes: scopes,
            requiredPathClasses: try require("required_path_classes"),
            scannedPathClasses: try require("scanned_path_classes"),
            scannedFileKinds: fileKinds,
            scannedPaths: try require("scanned_paths"),
            prohibitedHits: try require("prohibited_hits")
        )
    }

    static func releaseGateContracts(for module: Module) throws -> [SmeltCAMReleaseGateContract] {
        let descriptor = try SmeltCAMPackageDescriptor(from: registryModuleIR(module.fixture))
        let capabilities = try SmeltCAMPackageCapabilities(descriptor: descriptor)
        return try capabilities.releaseGateContracts(
            requiredGateIDs: module.requiredReleaseGates,
            releaseSurfaces: releaseSurfaceBindings(for: module)
        )
    }

    static func releaseContractIDs(
        for surface: SurfaceExpectation,
        in module: Module
    ) throws -> [String] {
        guard surface.requiresReleaseEvidence else { return [] }
        let descriptor = try SmeltCAMPackageDescriptor(from: registryModuleIR(module.fixture))
        let capabilities = try SmeltCAMPackageCapabilities(descriptor: descriptor)
        let contracts = try capabilities.releaseGateContracts(
            requiredGateIDs: module.requiredReleaseGates,
            releaseSurfaces: releaseSurfaceBindings(for: module)
        )
        return try capabilities.releaseContractIDs(
            for: releaseSurfaceBinding(surface),
            contracts: contracts
        )
    }

    private static func releaseSurfaceBindings(
        for module: Module
    ) -> [SmeltCAMReleaseSurfaceBinding] {
        module.surfaceExpectations.map(releaseSurfaceBinding)
    }

    private static func releaseSurfaceBinding(
        _ surface: SurfaceExpectation
    ) -> SmeltCAMReleaseSurfaceBinding {
        SmeltCAMReleaseSurfaceBinding(
            surfaceID: surface.surfaceID,
            exportID: surface.exportID,
            flowID: surface.flowID,
            selectedInputNames: surface.selectedInputNames,
            selectedInputs: surface.selectedInputs,
            selectedOutputNames: surface.selectedOutputNames,
            selectedOutputs: surface.selectedOutputs,
            gateIDs: surface.gateIDs,
            requiresReleaseEvidence: surface.requiresReleaseEvidence
        )
    }

    static func completionEvidenceDiagnostics(for module: Module) -> [String] {
        var diagnostics: [String] = []
        let surfaceIDs = module.surfaceExpectations.map(\.surfaceID)
        let surfaceIDSet = Set(surfaceIDs)
        if module.level >= .buildCommandCovered && module.surfaceExpectations.isEmpty {
            diagnostics.append("\(module.fixture) lacks target surface ledger")
        }
        if Set(surfaceIDs).count != surfaceIDs.count {
            diagnostics.append("\(module.fixture) has duplicate target surface IDs")
        }
        let incompleteSurfaces = module.surfaceExpectations.filter {
            $0.surfaceID.isEmpty
                || ($0.request != nil && (($0.exportID ?? "").isEmpty || ($0.flowID ?? "").isEmpty))
                || (!$0.gateIDs.isEmpty && $0.gateIDs.contains(where: \.isEmpty))
        }
        if !incompleteSurfaces.isEmpty {
            diagnostics.append("\(module.fixture) has incomplete target surface rows")
        }
        if module.requiredGates.contains("gate.first-audio")
            || module.requiredReleaseGates.contains("gate.first-audio")
            || module.requiredGates.contains("gate.startup-audio")
            || module.requiredReleaseGates.contains("gate.startup-audio")
            || module.surfaceExpectations.contains(where: {
                $0.gateIDs.contains("gate.first-audio")
                    || $0.gateIDs.contains("gate.startup-audio")
            }) {
            diagnostics.append("\(module.fixture) uses release surface label as CAM gate")
        }
        var surfacesByID: [String: SurfaceExpectation] = [:]
        for surface in module.surfaceExpectations where surfacesByID[surface.surfaceID] == nil {
            surfacesByID[surface.surfaceID] = surface
        }
        let commandObligations = commandSurfaceObligations(for: module)
        let obligationKeys = commandObligations.map {
            [$0.surfaceID, $0.command, $0.subcommand].joined(separator: "|")
        }
        if Set(obligationKeys).count != obligationKeys.count {
            diagnostics.append("\(module.fixture) has duplicate command surface obligations")
        }
        for obligation in commandObligations {
            guard let surface = surfacesByID[obligation.surfaceID] else {
                diagnostics.append("\(module.fixture) command obligation lacks surface coverage")
                continue
            }
            if surface.kind != .command {
                diagnostics.append("\(module.fixture) command obligation points at non-command surface")
            }
            if obligation.command == "trace" {
                if !obligation.surfaceID.hasPrefix("trace.") {
                    diagnostics.append("\(module.fixture) trace obligation points at non-trace surface")
                }
                if obligation.subcommand == "compare" {
                    diagnostics.append("\(module.fixture) trace compare cannot be package-opening coverage")
                }
            }
            if obligation.command == "linger-worker"
                && !obligation.surfaceID.hasPrefix("linger.") {
                diagnostics.append("\(module.fixture) linger obligation points at non-linger surface")
            }
            if obligation.bridgeAbsentRequired
                && (!surface.requiresRuntimeContract || !surface.requiresRuntimeAssembly) {
                diagnostics.append("\(module.fixture) command obligation overclaims bridge-free requirements")
            }
        }
        for surface in module.surfaceExpectations where surface.surfaceID.hasPrefix("trace.") {
            guard surface.requiresRuntimeContract || surface.requiresRuntimeAssembly else {
                continue
            }
            let subcommands = commandObligations
                .filter { $0.surfaceID == surface.surfaceID && $0.command == "trace" }
                .map(\.subcommand)
                .sorted()
            if subcommands != tracePackageOpeningSubcommands {
                diagnostics.append("\(module.fixture) trace surface lacks exact package-opening submodes")
            }
        }
        for surface in module.surfaceExpectations where surface.surfaceID.hasPrefix("linger.") {
            guard surface.requiresRuntimeContract || surface.requiresRuntimeAssembly else {
                continue
            }
            let subcommands = commandObligations
                .filter { $0.surfaceID == surface.surfaceID && $0.command == "linger-worker" }
                .map(\.subcommand)
                .sorted()
            if subcommands != ["worker"] {
                diagnostics.append("\(module.fixture) linger surface lacks worker delivery obligation")
            }
        }
        let optionalSurfaceIDs = module.publicOptionalInputs.map(\.surfaceID)
        if Set(optionalSurfaceIDs).count != optionalSurfaceIDs.count {
            diagnostics.append("\(module.fixture) has duplicate public optional input coverage")
        }
        for optionalInput in module.publicOptionalInputs {
            if optionalInput.defaultSemantics.isEmpty {
                diagnostics.append("\(module.fixture) public optional input lacks default semantics")
            }
            if optionalInput.acceptedValueSemantics.isEmpty {
                diagnostics.append("\(module.fixture) public optional input lacks accepted-value semantics")
            }
            if optionalInput.closedSurfaceEvidenceToken.isEmpty {
                diagnostics.append("\(module.fixture) public optional input lacks closed negative evidence")
            }
            if optionalInput.negativeSemantics.isEmpty {
                diagnostics.append("\(module.fixture) public optional input lacks negative-value semantics")
            }
            guard let surface = surfacesByID[optionalInput.surfaceID] else {
                diagnostics.append("\(module.fixture) public optional input lacks surface coverage")
                continue
            }
            if surface.kind != .optionalInputCoverage
                || surface.exportID != optionalInput.exportID
                || surface.flowID != optionalInput.flowID
                || surface.gateIDs != optionalInput.gateIDs
                || !surface.selectedInputs.isEmpty
                || !surface.selectedOutputs.isEmpty
                || surface.requiresReleaseEvidence
                || surface.requiresRuntimeContract
                || surface.requiresRuntimeAssembly {
                diagnostics.append("\(module.fixture) has invalid public optional input coverage")
            }
            let hasClosedBlocker = module.closedSurfaces.contains {
                $0.surfaceID == optionalInput.surfaceID
            }
            let optionalInputIsNative = surface.adapterBoundary == .none
            if !optionalInputIsNative && !hasClosedBlocker {
                diagnostics.append("\(module.fixture) public optional input lacks closed surface blocker")
            }
            let hasClosedNegativeEvidence = module.closedSurfaces.contains {
                $0.surfaceID == optionalInput.surfaceID
                    && $0.reason.contains(optionalInput.closedSurfaceEvidenceToken)
            }
            if !optionalInputIsNative && !hasClosedNegativeEvidence {
                diagnostics.append("\(module.fixture) public optional input lacks closed negative evidence")
            }
        }
        let ambiguousInputSurfaces = module.surfaceExpectations.filter {
            hasDuplicatePortShapes($0.selectedInputs)
                && !hasCompletePortNames($0.selectedInputNames, count: $0.selectedInputs.count)
        }
        if !ambiguousInputSurfaces.isEmpty {
            diagnostics.append("\(module.fixture) has duplicate selected input shapes without explicit port names")
        }
        let ambiguousOutputSurfaces = module.surfaceExpectations.filter {
            hasDuplicatePortShapes($0.selectedOutputs)
                && !hasCompletePortNames($0.selectedOutputNames, count: $0.selectedOutputs.count)
        }
        if !ambiguousOutputSurfaces.isEmpty {
            diagnostics.append("\(module.fixture) has duplicate selected output shapes without explicit port names")
        }
        if module.level >= .capabilityResolved && module.capabilityExpectations.isEmpty {
            diagnostics.append("\(module.fixture) lacks capability expectations")
        }
        if module.level < .capabilityResolved && !module.capabilityExpectations.isEmpty {
            diagnostics.append("\(module.fixture) overclaims capability resolution")
        }
        let ambiguousInputCapabilities = module.capabilityExpectations.filter {
            hasDuplicatePortShapes($0.selectedInputs)
                && !hasCompletePortNames($0.selectedInputNames, count: $0.selectedInputs.count)
        }
        if !ambiguousInputCapabilities.isEmpty {
            diagnostics.append("\(module.fixture) has duplicate capability input shapes without explicit port names")
        }
        let ambiguousOutputCapabilities = module.capabilityExpectations.filter {
            hasDuplicatePortShapes($0.selectedOutputs)
                && !hasCompletePortNames($0.selectedOutputNames, count: $0.selectedOutputs.count)
        }
        if !ambiguousOutputCapabilities.isEmpty {
            diagnostics.append("\(module.fixture) has duplicate capability output shapes without explicit port names")
        }
        if module.level >= .checkedPackageProjected && module.packageProjection == nil {
            diagnostics.append("\(module.fixture) lacks checked package projection facts")
        }
        if module.level < .checkedPackageProjected && module.packageProjection != nil {
            diagnostics.append("\(module.fixture) overclaims checked projection")
        }
        let undeclaredReleaseGates = module.requiredReleaseGates.filter {
            !module.requiredGates.contains($0)
        }
        if !undeclaredReleaseGates.isEmpty {
            diagnostics.append(
                "\(module.fixture) release gates are not declared required gates: \(undeclaredReleaseGates.joined(separator: ","))"
            )
        }
        if module.level >= .releaseGated {
            if module.requiredReleaseGates.isEmpty {
                diagnostics.append("\(module.fixture) lacks required CAM release gates")
            }
            if module.releaseGateEvidence.isEmpty {
                diagnostics.append("\(module.fixture) lacks release gate evidence")
            }
            let incompleteReleaseEvidence = module.releaseGateEvidence.filter {
                let requiresEventEndpoints = releaseEvidenceRequiresEventEndpoints($0)
                return $0.evidenceID.isEmpty
                    || $0.surfaceID.isEmpty
                    || ($0.request != nil && ($0.exportID.isEmpty || $0.flowID.isEmpty))
                    || $0.selectedCapability.isEmpty
                    || $0.gateID.isEmpty
                    || $0.gateTraceLabels.isEmpty
                    || $0.gateTraceLabels.contains(where: \.isEmpty)
                    || $0.metric.isEmpty
                    || $0.metricPath.isEmpty
                    || $0.comparator.isEmpty
                    || $0.bound.isEmpty
                    || (requiresEventEndpoints && ($0.fromEventID.isEmpty || $0.toEventID.isEmpty))
                    || $0.processMode.isEmpty
                    || $0.processStartTimestamp.isEmpty
                    || $0.cacheState.isEmpty
                    || $0.camSemanticSHA256.isEmpty
                    || $0.exportABISHA256.isEmpty
                    || $0.camDescriptorSHA256.isEmpty
                    || $0.descriptorGraphSignatureSHA256.isEmpty
                    || $0.packageProjectionID.isEmpty
                    || $0.packageProjectionVersion <= 0
                    || $0.projectedPackageSpecSHA256.isEmpty
                    || $0.packageFiles.isEmpty
                    || $0.packageFiles.contains(where: \.isEmpty)
                    // build-evidence + package-payload pins are a disablable PAIR
                    // ("" = disabled; builds are not byte-reproducible). Both-empty
                    // is allowed; only a MIXED state (exactly one empty) is incomplete.
                    || ($0.buildEvidenceSHA256.isEmpty != $0.packagePayloadSHA256.isEmpty)
                    || $0.commandStamp.isEmpty
                    || $0.measurementSHA256.isEmpty
                    || $0.provenanceSHA256.isEmpty
            }
            if !incompleteReleaseEvidence.isEmpty {
                diagnostics.append("\(module.fixture) has incomplete release gate evidence")
            }
            let invalidReleaseShape = module.releaseGateEvidence.filter {
                !isReleaseProcessMode($0.processMode)
                    || !isReleaseCacheState($0.cacheState)
                    || !isISO8601Timestamp($0.processStartTimestamp)
                    || !metricPathCoversGateEvidence($0)
            }
            if !invalidReleaseShape.isEmpty {
                diagnostics.append("\(module.fixture) has invalid release gate evidence shape")
            }
            let invalidReleaseHashes = module.releaseGateEvidence.filter {
                !isSHA256Hex($0.camSemanticSHA256)
                    || !isSHA256Hex($0.exportABISHA256)
                    || !isSHA256Hex($0.camDescriptorSHA256)
                    || !isSHA256Hex($0.descriptorGraphSignatureSHA256)
                    || !isSHA256Hex($0.projectedPackageSpecSHA256)
                    || !isDisabledOrSHA256Hex($0.buildEvidenceSHA256)
                    || !isDisabledOrSHA256Hex($0.packagePayloadSHA256)
                    || !isSHA256Hex($0.measurementSHA256)
                    || !isSHA256Hex($0.provenanceSHA256)
            }
            if !invalidReleaseHashes.isEmpty {
                diagnostics.append("\(module.fixture) has invalid release gate evidence hashes")
            }
            let mismatchedReleaseIdentity = module.releaseGateEvidence.filter {
                !hashesMatch($0.camSemanticSHA256, module.semanticSHA256)
                    || !hashesMatch($0.exportABISHA256, module.exportABISHA256)
                    || !hashesMatch(
                        $0.descriptorGraphSignatureSHA256,
                        module.descriptorGraphSignatureSHA256
                    )
                    || !releasePackageIdentityMatches($0, module.packageProjection)
            }
            if !mismatchedReleaseIdentity.isEmpty {
                diagnostics.append("\(module.fixture) release gate evidence identity does not match module identity")
            }
            let selectorShapedReleaseEvidence = module.releaseGateEvidence.filter {
                releaseEvidenceIdentityFields($0).contains { field in
                    releaseEvidenceBannedSelectorTerms.contains { banned in
                        field.localizedCaseInsensitiveContains(banned)
                    }
                }
            }
            if !selectorShapedReleaseEvidence.isEmpty {
                diagnostics.append("\(module.fixture) has selector-shaped release gate evidence")
            }
            let releaseEvidenceIDs = module.releaseGateEvidence.map(\.evidenceID)
            if Set(releaseEvidenceIDs).count != releaseEvidenceIDs.count {
                diagnostics.append("\(module.fixture) has duplicate release gate evidence IDs")
            }
            let unknownReleaseSurfaces = module.releaseGateEvidence.filter {
                !surfaceIDSet.contains($0.surfaceID)
            }
            if !unknownReleaseSurfaces.isEmpty {
                diagnostics.append("\(module.fixture) has release evidence for unknown surfaces")
            }
            let mismatchedReleaseSurfaceContracts = module.releaseGateEvidence.filter {
                guard let surface = surfacesByID[$0.surfaceID] else { return false }
                if surface.exportID != nil && surface.exportID != $0.exportID { return true }
                if surface.flowID != nil && surface.flowID != $0.flowID { return true }
                if let request = surface.request, request != $0.request { return true }
                return !surface.gateIDs.contains($0.gateID)
            }
            if !mismatchedReleaseSurfaceContracts.isEmpty {
                diagnostics.append("\(module.fixture) release gate evidence does not match release surface contracts")
            }
            let inventoryBackedCorrectnessReleaseEvidence = module.releaseGateEvidence.filter {
                guard surfacesByID[$0.surfaceID]?.kind == .correctness else { return false }
                return $0.metric == "package-files" || $0.metricPath.hasPrefix("inventory:")
            }
            if !inventoryBackedCorrectnessReleaseEvidence.isEmpty {
                diagnostics.append("\(module.fixture) correctness release evidence is backed by inventory evidence")
            }
            let releaseEvidenceSurfaceIDs = Set(module.releaseGateEvidence.map(\.surfaceID))
            let requiredReleaseSurfaceIDs = module.surfaceExpectations
                .filter(\.requiresReleaseEvidence)
                .map(\.surfaceID)
            if releaseEvidenceSurfaceIDs != Set(requiredReleaseSurfaceIDs) {
                diagnostics.append(
                    "\(module.fixture) release gate evidence does not cover every required release surface"
                )
            }
            let releaseEvidenceGateIDsBySurface = Dictionary(
                grouping: module.releaseGateEvidence,
                by: \.surfaceID
            ).mapValues { Set($0.map(\.gateID)) }
            let releaseSurfacesWithUncoveredGateContracts = module.surfaceExpectations
                .filter(\.requiresReleaseEvidence)
                .filter { surface in
                    releaseEvidenceGateIDsBySurface[surface.surfaceID] != Set(surface.gateIDs)
                }
            if !releaseSurfacesWithUncoveredGateContracts.isEmpty {
                diagnostics.append(
                    "\(module.fixture) release gate evidence does not cover every gate for each required release surface"
                )
            }
            let releaseEvidenceGateIDs = Set(module.releaseGateEvidence.map(\.gateID))
            if releaseEvidenceGateIDs != Set(module.requiredReleaseGates) {
                diagnostics.append(
                    "\(module.fixture) release gate evidence does not cover every required CAM release gate"
                )
            }
            if module.releaseGateEvidence.contains(where: {
                $0.gateID == "gate.first-audio" || $0.gateID == "gate.startup-audio"
            }) {
                diagnostics.append("\(module.fixture) uses release surface label as release gate")
            }
            let releaseLabelFields = module.releaseGateEvidence.filter { evidence in
                let fields = [
                    evidence.evidenceID,
                    evidence.selectedCapability,
                    evidence.gateID,
                    evidence.metric,
                    evidence.metricPath,
                    evidence.fromEventID,
                    evidence.toEventID,
                    evidence.commandStamp,
                ] + evidence.gateTraceLabels
                return fields.contains {
                    $0.contains("gate.first-audio") || $0.contains("gate.startup-audio")
                }
            }
            if !releaseLabelFields.isEmpty {
                diagnostics.append("\(module.fixture) uses release surface label as release evidence field")
            }
            let adapterBoundaryReleaseSurfaces = module.surfaceExpectations.filter {
                $0.requiresReleaseEvidence && $0.adapterBoundary != .none
            }
            if !adapterBoundaryReleaseSurfaces.isEmpty {
                diagnostics.append("\(module.fixture) claims release gating with adapter-boundary release surfaces")
            }
            let adapterBoundaryBuildSurfaces = module.surfaceExpectations.filter {
                $0.kind == .build && $0.adapterBoundary != .none
            }
            if !adapterBoundaryBuildSurfaces.isEmpty {
                diagnostics.append("\(module.fixture) claims release gating with adapter-boundary build surfaces")
            }
        } else if !module.releaseGateEvidence.isEmpty {
            diagnostics.append("\(module.fixture) overclaims release gating")
        }
        if module.level >= .runtimeContractCovered {
            if module.runtimeContractEvidence.isEmpty {
                diagnostics.append("\(module.fixture) lacks runtime contract evidence")
            }
            let incomplete = module.runtimeContractEvidence.filter {
                $0.bodySHA256.isEmpty || $0.provenanceSHA256.isEmpty
            }
            if !incomplete.isEmpty {
                diagnostics.append("\(module.fixture) has incomplete runtime contract evidence")
            }
            let evidenceIDs = module.runtimeContractEvidence.map(\.evidenceID)
            if Set(evidenceIDs).count != evidenceIDs.count {
                diagnostics.append("\(module.fixture) has duplicate runtime contract evidence IDs")
            }
            let unknownRuntimeSurfaces = module.runtimeContractEvidence.filter {
                !surfaceIDSet.contains($0.surfaceID)
            }
            if !unknownRuntimeSurfaces.isEmpty {
                diagnostics.append("\(module.fixture) has runtime contract evidence for unknown surfaces")
            }
            let runtimeEvidenceSurfaceIDs = module.runtimeContractEvidence.map(\.surfaceID)
            if Set(runtimeEvidenceSurfaceIDs).count != runtimeEvidenceSurfaceIDs.count {
                diagnostics.append("\(module.fixture) has duplicate runtime contract evidence surfaces")
            }
            let requiredRuntimeSurfaceIDs = module.surfaceExpectations
                .filter(\.requiresRuntimeContract)
                .map(\.surfaceID)
                .sorted()
            if runtimeEvidenceSurfaceIDs.sorted() != requiredRuntimeSurfaceIDs {
                diagnostics.append(
                    "\(module.fixture) runtime contract evidence does not cover every required surface"
                )
            }
            if runtimeContractCoverageKeys(module.runtimeContractEvidence)
                != runtimeContractCoverageKeys(runtimeRequiredCapabilityExpectations(for: module)) {
                diagnostics.append(
                    "\(module.fixture) runtime contract evidence does not cover every runtime-required capability expectation"
                )
            }
            let adapterBoundaryRuntimeSurfaces = module.surfaceExpectations.filter {
                $0.requiresRuntimeContract && $0.adapterBoundary != .none
            }
            if !adapterBoundaryRuntimeSurfaces.isEmpty {
                diagnostics.append(
                    "\(module.fixture) claims runtime contract coverage with adapter-boundary runtime surfaces"
                )
            }
        }
        if module.level >= .runtimeAssemblyCovered {
            if module.runtimeAssemblyEvidence.isEmpty {
                diagnostics.append("\(module.fixture) lacks runtime assembly evidence")
            }
            let incompleteRuntimeAssembly = module.runtimeAssemblyEvidence.filter {
                $0.surfaceID.isEmpty
                    || $0.evidenceID.isEmpty
                    || $0.exportID.isEmpty
                    || $0.flowID.isEmpty
                    || $0.assemblyPlanKey.isEmpty
                    || $0.executionBackend.isEmpty
                    || $0.artifactRoles.isEmpty
                    || $0.gateIDs.isEmpty
                    || $0.selectorDeletionLintID.isEmpty
            }
            if !incompleteRuntimeAssembly.isEmpty {
                diagnostics.append("\(module.fixture) has incomplete runtime assembly evidence")
            }
            let incompleteFeatureSets = module.runtimeAssemblyEvidence.filter {
                $0.requiredFeatureSet.isEmpty
                    || $0.consumedFeatureSet.isEmpty
                    || $0.requiredFeatureSet.contains(where: \.isEmpty)
                    || $0.consumedFeatureSet.contains(where: \.isEmpty)
                    || $0.unsupportedFeatureSet.contains(where: \.isEmpty)
                    || Set($0.requiredFeatureSet).count != $0.requiredFeatureSet.count
                    || Set($0.consumedFeatureSet).count != $0.consumedFeatureSet.count
                    || Set($0.unsupportedFeatureSet).count != $0.unsupportedFeatureSet.count
            }
            if !incompleteFeatureSets.isEmpty {
                diagnostics.append("\(module.fixture) has invalid runtime assembly feature sets")
            }
            let incompleteObligationIDs = module.runtimeAssemblyEvidence.filter {
                $0.requiredObligationIDs.contains(where: { !isSHA256Hex($0) })
                    || $0.consumedObligationIDs.contains(where: { !isSHA256Hex($0) })
                    || $0.unsupportedObligationIDs.contains(where: { !isSHA256Hex($0) })
                    || Set($0.requiredObligationIDs).count != $0.requiredObligationIDs.count
                    || Set($0.consumedObligationIDs).count != $0.consumedObligationIDs.count
                    || Set($0.unsupportedObligationIDs).count != $0.unsupportedObligationIDs.count
                    || $0.requiredObligationIDs != $0.requiredObligationIDs.sorted()
                    || $0.consumedObligationIDs != $0.consumedObligationIDs.sorted()
                    || $0.unsupportedObligationIDs != $0.unsupportedObligationIDs.sorted()
            }
            if !incompleteObligationIDs.isEmpty {
                diagnostics.append("\(module.fixture) has invalid runtime assembly obligation ids")
            }
            let selectorShapedRuntimeAssemblyEvidence = module.runtimeAssemblyEvidence.filter {
                runtimeAssemblyEvidenceIdentityFields($0).contains { field in
                    runtimeAssemblyEvidenceBannedSelectorTerms.contains { banned in
                        field.localizedCaseInsensitiveContains(banned)
                    }
                }
            }
            if !selectorShapedRuntimeAssemblyEvidence.isEmpty {
                diagnostics.append("\(module.fixture) has selector-shaped runtime assembly evidence")
            }
            let unsupportedRuntimeAssembly = module.runtimeAssemblyEvidence.filter {
                !$0.unsupportedFeatureSet.isEmpty
            }
            if !unsupportedRuntimeAssembly.isEmpty {
                diagnostics.append("\(module.fixture) claims runtime assembly with unsupported feature sets")
            }
            let unsupportedRuntimeAssemblyObligations = module.runtimeAssemblyEvidence.filter {
                !$0.unsupportedObligationIDs.isEmpty
            }
            if !unsupportedRuntimeAssemblyObligations.isEmpty {
                diagnostics.append("\(module.fixture) claims runtime assembly with unsupported obligation ids")
            }
            let unconsumedFeatureSets = module.runtimeAssemblyEvidence.filter {
                Set($0.requiredFeatureSet).subtracting($0.consumedFeatureSet)
                    != Set($0.unsupportedFeatureSet)
            }
            if !unconsumedFeatureSets.isEmpty {
                diagnostics.append("\(module.fixture) runtime assembly feature sets do not balance")
            }
            let unconsumedObligationIDs = module.runtimeAssemblyEvidence.filter {
                Set($0.requiredObligationIDs).subtracting($0.consumedObligationIDs)
                    != Set($0.unsupportedObligationIDs)
                    || !Set($0.consumedObligationIDs).isSubset(of: Set($0.requiredObligationIDs))
                    || !Set($0.unsupportedObligationIDs).isSubset(of: Set($0.requiredObligationIDs))
            }
            if !unconsumedObligationIDs.isEmpty {
                diagnostics.append("\(module.fixture) runtime assembly obligation ids do not balance")
            }
            let unknownRuntimeAssemblySurfaces = module.runtimeAssemblyEvidence.filter {
                !surfaceIDSet.contains($0.surfaceID)
            }
            if !unknownRuntimeAssemblySurfaces.isEmpty {
                diagnostics.append("\(module.fixture) has runtime assembly evidence for unknown surfaces")
            }
            let runtimeAssemblyEvidenceSurfaceIDs = module.runtimeAssemblyEvidence.map(\.surfaceID)
            if Set(runtimeAssemblyEvidenceSurfaceIDs).count != runtimeAssemblyEvidenceSurfaceIDs.count {
                diagnostics.append("\(module.fixture) has duplicate runtime assembly evidence surfaces")
            }
            let requiredRuntimeAssemblySurfaceIDs = module.surfaceExpectations
                .filter(\.requiresRuntimeAssembly)
                .map(\.surfaceID)
                .sorted()
            if runtimeAssemblyEvidenceSurfaceIDs.sorted() != requiredRuntimeAssemblySurfaceIDs {
                diagnostics.append(
                    "\(module.fixture) runtime assembly evidence does not cover every required surface"
                )
            }
            if module.runtimeAssemblyEvidence.contains(where: { !$0.manifestPolicyBridgeAbsent }) {
                diagnostics.append("\(module.fixture) runtime assembly evidence crosses manifest bridge")
            }
        }
        if module.level >= .selectorRootsDeleted {
            if module.selectorDeletionScans.isEmpty {
                diagnostics.append("\(module.fixture) lacks selector deletion scan evidence")
            }
            let runtimeAssemblyLintIDs = Set(
                module.runtimeAssemblyEvidence
                    .map(\.selectorDeletionLintID)
                    .filter { !$0.isEmpty }
            )
            let scanLintIDs = Set(module.selectorDeletionScans.map(\.lintID))
            if scanLintIDs != runtimeAssemblyLintIDs {
                diagnostics.append(
                    "\(module.fixture) selector deletion scans do not match assembly lint IDs"
                )
            }
            let duplicateScanLintIDs = duplicateSelectorDeletionScanLintIDs(
                module.selectorDeletionScans
            )
            if !duplicateScanLintIDs.isEmpty {
                diagnostics.append("\(module.fixture) has duplicate selector deletion scan IDs")
            }
            let incompleteScans = module.selectorDeletionScans.filter {
                $0.lintID.isEmpty
                    || $0.scannerVersion != selectorDeletionScannerVersion
                    || $0.command.isEmpty
                    || $0.command.contains(where: \.isEmpty)
                    || !isSHA256Hex($0.sourceRootSHA256)
                    || !isSHA256Hex($0.allowlistSHA256)
                    || !isSHA256Hex($0.provenanceSHA256)
                    || $0.requiredPathClasses.isEmpty
                    || $0.scannedPathClasses.isEmpty
                    || $0.scannedFileKinds.isEmpty
                    || $0.scannedPaths.isEmpty
            }
            if !incompleteScans.isEmpty {
                diagnostics.append("\(module.fixture) has incomplete selector deletion scan evidence")
            }
            let requiredScopes = Set(ProductionSelectorScope.allCases)
            let incompleteScopeScans = module.selectorDeletionScans.filter {
                Set($0.scopes) != requiredScopes || Set($0.scopes).count != $0.scopes.count
            }
            if !incompleteScopeScans.isEmpty {
                diagnostics.append(
                    "\(module.fixture) lacks complete production selector scope coverage"
                )
            }
            let requiredPathClassSet = Set(selectorDeletionRequiredPathClasses)
            let allowedPathClassSet = requiredPathClassSet.union(selectorDeletionOptionalPathClasses)
            let incompletePathClassScans = module.selectorDeletionScans.filter { scan in
                Set(scan.requiredPathClasses) != requiredPathClassSet
                    || Set(scan.requiredPathClasses).count != scan.requiredPathClasses.count
                    || !requiredPathClassSet.isSubset(of: Set(scan.scannedPathClasses))
                    || !Set(scan.scannedPathClasses).isSubset(of: allowedPathClassSet)
                    || Set(scan.scannedPathClasses).count != scan.scannedPathClasses.count
            }
            if !incompletePathClassScans.isEmpty {
                diagnostics.append(
                    "\(module.fixture) lacks complete selector deletion path class coverage"
                )
            }
            let invalidPathScans = module.selectorDeletionScans.filter { scan in
                scan.scannedPaths.contains { !isProductionSelectorScanPath($0) }
            }
            if !invalidPathScans.isEmpty {
                diagnostics.append("\(module.fixture) has invalid selector deletion scan paths")
            }
            let pathClassMismatchScans = module.selectorDeletionScans.filter { scan in
                let classesFromPaths = Set(scan.scannedPaths.map(selectorDeletionPathClass))
                    .subtracting(["unknown"])
                return classesFromPaths != Set(scan.scannedPathClasses)
            }
            if !pathClassMismatchScans.isEmpty {
                diagnostics.append(
                    "\(module.fixture) selector deletion path classes do not match scanned paths"
                )
            }
            let fileKindMismatchScans = module.selectorDeletionScans.filter { scan in
                var expectedKindCounts: [String: Int] = [:]
                for path in scan.scannedPaths where isProductionSelectorScanPath(path) {
                    let kind = selectorDeletionFileKind(path)
                    expectedKindCounts[kind, default: 0] += 1
                }
                var actualKindCounts: [String: Int] = [:]
                for fileKind in scan.scannedFileKinds {
                    guard !fileKind.kind.isEmpty, fileKind.count > 0 else { return true }
                    actualKindCounts[fileKind.kind] = fileKind.count
                }
                return expectedKindCounts != actualKindCounts
            }
            if !fileKindMismatchScans.isEmpty {
                diagnostics.append(
                    "\(module.fixture) selector deletion file kind counts do not match scanned paths"
                )
            }
            let selectorShapedScans = module.selectorDeletionScans.filter { scan in
                selectorDeletionEvidenceIdentityFields(scan).contains { field in
                    containsSelectorDeletionBannedTerm(field)
                }
            }
            if !selectorShapedScans.isEmpty {
                diagnostics.append("\(module.fixture) has selector-shaped selector deletion evidence")
            }
            if module.selectorDeletionScans.contains(where: { !$0.prohibitedHits.isEmpty }) {
                diagnostics.append("\(module.fixture) selector deletion scan found prohibited selector roots")
            }
        } else if !module.selectorDeletionScans.isEmpty {
            diagnostics.append("\(module.fixture) overclaims selector root deletion")
        }
        let closedSurfaceIssues = module.closedSurfaces.filter {
            $0.surfaceID.isEmpty || $0.surface.isEmpty || $0.reason.isEmpty
        }
        if !closedSurfaceIssues.isEmpty {
            diagnostics.append("\(module.fixture) has incomplete closed surface blockers")
        }
        let danglingClosedSurfaces = module.closedSurfaces.filter {
            !surfaceIDSet.contains($0.surfaceID)
        }
        if !danglingClosedSurfaces.isEmpty {
            diagnostics.append("\(module.fixture) has closed surface blockers for unknown surfaces")
        }
        let closedSurfaceNames = module.closedSurfaces.map(\.surface)
        if Set(closedSurfaceNames).count != closedSurfaceNames.count {
            diagnostics.append("\(module.fixture) has duplicate closed surface blockers")
        }
        let closedSurfaceIDs = module.closedSurfaces.map(\.surfaceID)
        if Set(closedSurfaceIDs).count != closedSurfaceIDs.count {
            diagnostics.append("\(module.fixture) has duplicate closed surface blocker IDs")
        }
        let closedSurfaceIDSet = Set(closedSurfaceIDs)
        let unblockedAdapterBoundarySurfaces = module.surfaceExpectations.filter {
            $0.kind != .build
                && $0.adapterBoundary != .none
                && !closedSurfaceIDSet.contains($0.surfaceID)
        }
        if !unblockedAdapterBoundarySurfaces.isEmpty {
            diagnostics.append("\(module.fixture) has adapter-boundary surfaces without closed surface blockers")
        }
        if module.level >= .capabilityRouted && !module.closedSurfaces.isEmpty {
            diagnostics.append("\(module.fixture) claims capability routing with closed surfaces")
        }
        let adapterBoundarySurfaces = module.surfaceExpectations.filter {
            $0.adapterBoundary != .none
        }
        if module.level >= .capabilityRouted && !adapterBoundarySurfaces.isEmpty {
            diagnostics.append(
                "\(module.fixture) claims capability routing with adapter-boundary surfaces"
            )
        }
        return diagnostics
    }

    private static func isSHA256Hex(_ value: String) -> Bool {
        value.utf8.count == 64
            && value.utf8.allSatisfy { byte in
                (byte >= 48 && byte <= 57)
                    || (byte >= 97 && byte <= 102)
                    || (byte >= 65 && byte <= 70)
            }
    }

    /// A build-evidence / package-payload pin is either a valid sha256 or the
    /// empty disabled sentinel (""). Builds are not byte-reproducible, so these
    /// two content pins are disabled together (see the paired isEmpty guard in
    /// the completeness scan and releasePackageIdentityMatches, which matches
    /// "" against "").
    private static func isDisabledOrSHA256Hex(_ value: String) -> Bool {
        value.isEmpty || isSHA256Hex(value)
    }

    private static func hashesMatch(_ lhs: String, _ rhs: String) -> Bool {
        lhs.lowercased() == rhs.lowercased()
    }

    private static func releasePackageIdentityMatches(
        _ evidence: ReleaseGateExpectation,
        _ projection: PackageProjectionExpectation?
    ) -> Bool {
        guard let projection,
              let releaseBuildEvidenceSHA256 = projection.releaseBuildEvidenceSHA256,
              let releasePackagePayloadSHA256 = projection.releasePackagePayloadSHA256
        else { return false }

        return evidence.packageProjectionID == projection.id
            && evidence.packageProjectionVersion == projection.version
            && hashesMatch(
                evidence.projectedPackageSpecSHA256,
                projection.projectedPackageSpecSHA256
            )
            && evidence.packageFiles == projection.packageFiles
            && hashesMatch(evidence.buildEvidenceSHA256, releaseBuildEvidenceSHA256)
            && hashesMatch(evidence.packagePayloadSHA256, releasePackagePayloadSHA256)
    }

    private static func isISO8601Timestamp(_ value: String) -> Bool {
        guard value.hasSuffix("Z") else { return false }
        return ISO8601DateFormatter().date(from: value) != nil
    }

    private static func isReleaseProcessMode(_ value: String) -> Bool {
        ["process-cold", "process-warm"].contains(value)
    }

    private static func isReleaseCacheState(_ value: String) -> Bool {
        [
            "process-cold",
            "process-warm",
            "host-global-cold-candidate",
            "warm",
            "cold",
        ].contains(value)
    }

    private static func releaseEvidenceRequiresEventEndpoints(_ evidence: ReleaseGateExpectation) -> Bool {
        nonEventMetricPathPrefix(for: evidence) == nil
    }

    private static func metricPathCoversGateEvidence(_ evidence: ReleaseGateExpectation) -> Bool {
        if let prefix = nonEventMetricPathPrefix(for: evidence) {
            return evidence.metricPath == "\(prefix):gate.\(evidence.gateID):\(evidence.metric)"
        }
        return evidence.metricPath.contains(evidence.fromEventID)
            && evidence.metricPath.contains(evidence.toEventID)
            && evidence.metricPath.contains(evidence.metric)
    }

    private static func nonEventMetricPathPrefix(for evidence: ReleaseGateExpectation) -> String? {
        if evidence.metricPath.hasPrefix("scalar-metric:") {
            return "scalar-metric"
        }
        if evidence.metricPath.hasPrefix("inventory:") {
            return "inventory"
        }
        if evidence.metricPath.hasPrefix("evidence:") {
            return "evidence"
        }
        return nil
    }

    private static let releaseEvidenceBannedSelectorTerms: [String] = [
        "qwen",
        "qwen-tts",
        "llm",
        "tts",
        "asr",
        "ttft",
        "ttfa",
        "family",
        "kind",
        "arch",
        "architecture",
        "modality",
        "profile",
        "bucket",
        "target",
    ]

    private static func releaseEvidenceIdentityFields(
        _ evidence: ReleaseGateExpectation
    ) -> [String] {
        var fields = [
            evidence.surfaceID,
            evidence.evidenceID,
            evidence.exportID,
            evidence.flowID,
            evidence.selectedCapability,
            evidence.gateID,
            evidence.metric,
            evidence.metricPath,
            evidence.comparator,
            evidence.unit ?? "",
            evidence.fromEventID,
            evidence.toEventID,
            evidence.processMode,
            evidence.cacheState,
            evidence.camSemanticSHA256,
            evidence.exportABISHA256,
            evidence.camDescriptorSHA256,
            evidence.descriptorGraphSignatureSHA256,
            evidence.commandStamp,
        ] + evidence.gateTraceLabels
        if evidence.metric != "package-files" {
            fields.append(evidence.bound)
        }
        return fields
    }

    private static let runtimeAssemblyEvidenceBannedSelectorTerms: [String] = [
        "qwen",
        "qwen-tts",
        "llm",
        "tts",
        "asr",
        "family",
        "kind",
        "arch",
        "architecture",
        "modality",
        "profile",
        "bucket",
        "target",
    ]

    private static let selectorDeletionEvidenceBannedSelectorTerms: [String] = [
        "qwen",
        "qwen-tts",
        "whisper",
        "llm",
        "tts",
        "asr",
        "family",
        "kind",
        "arch",
        "architecture",
        "modality",
        "profile",
        "bucket",
        "target",
        "policy-mode",
        "manifest-bridge",
        "package-kind",
    ]

    static let selectorDeletionRequiredPathClasses: [String] = [
        "source",
        "tool_script",
        "package_manifest",
        "workflow",
        "local_workflow_action",
        "generated_module_catalog",
        "module_plan",
    ]

    private static let selectorDeletionOptionalPathClasses: [String] = [
        "public_launcher",
    ]

    private static func duplicateSelectorDeletionScanLintIDs(
        _ scans: [SelectorDeletionScanExpectation]
    ) -> [String] {
        var seen: Set<String> = []
        var duplicates: Set<String> = []
        for lintID in scans.map(\.lintID) {
            if seen.contains(lintID) {
                duplicates.insert(lintID)
            }
            seen.insert(lintID)
        }
        return duplicates.sorted()
    }

    private static func isProductionSelectorScanPath(_ path: String) -> Bool {
        guard !path.isEmpty, !path.hasPrefix("/"), !path.contains("\\") else {
            return false
        }
        let components = path.split(separator: "/", omittingEmptySubsequences: false)
        guard !components.contains(".."), !components.contains(".") else {
            return false
        }
        return selectorDeletionPathClass(path) != "unknown"
    }

    private static func selectorDeletionPathClass(_ path: String) -> String {
        if path.hasPrefix("Sources/") { return "source" }
        if path == "Package.swift" { return "package_manifest" }
        if path.hasPrefix("tools/") { return "tool_script" }
        if path.hasPrefix(".github/workflows/") { return "workflow" }
        if path.hasPrefix(".github/actions/") { return "local_workflow_action" }
        if path == "Models/completion-matrix.release-surfaces.json" {
            return "generated_module_catalog"
        }
        if path == "Models/source-build-plan.tsv"
            || path == "Models/package-build-plan.tsv" {
            return "module_plan"
        }
        if path.hasPrefix("bin/") || path.hasPrefix("scripts/") {
            return "public_launcher"
        }
        return "unknown"
    }

    private static func selectorDeletionFileKind(_ path: String) -> String {
        if path == "Package.swift" || path.hasSuffix(".swift") { return "swift" }
        if path.hasSuffix(".yml") || path.hasSuffix(".yaml") { return "yaml" }
        if path.hasSuffix(".json") { return "json" }
        if path.hasSuffix(".sh") { return "shell" }
        let fileName = path.split(separator: "/").last.map(String.init) ?? path
        let extensionPart: String
        if let dotIndex = fileName.lastIndex(of: "."), dotIndex != fileName.startIndex {
            extensionPart = String(fileName[fileName.index(after: dotIndex)...])
        } else {
            extensionPart = ""
        }
        if path.hasSuffix(".py") || path.hasPrefix("tools/") && extensionPart.isEmpty {
            return "tool"
        }
        if ["js", "mjs", "ts"].contains(extensionPart) { return "javascript" }
        if extensionPart == "rb" { return "ruby" }
        return extensionPart.isEmpty ? "extensionless" : extensionPart
    }

    private static func selectorDeletionEvidenceIdentityFields(
        _ scan: SelectorDeletionScanExpectation
    ) -> [String] {
        [scan.lintID]
    }

    private static func containsSelectorDeletionBannedTerm(_ value: String) -> Bool {
        let candidates = selectorDeletionTermCandidates(value)
        return selectorDeletionEvidenceBannedSelectorTerms.contains { banned in
            let bannedCandidates = selectorDeletionTermCandidates(banned)
            return !candidates.isDisjoint(with: bannedCandidates)
        }
    }

    private static func selectorDeletionTermCandidates(_ value: String) -> Set<String> {
        var expanded = ""
        var previousWasLowercaseOrDigit = false
        for scalar in value.unicodeScalars {
            let isUppercase = CharacterSet.uppercaseLetters.contains(scalar)
            if isUppercase && previousWasLowercaseOrDigit {
                expanded.append("-")
            }
            expanded.unicodeScalars.append(scalar)
            previousWasLowercaseOrDigit =
                CharacterSet.lowercaseLetters.contains(scalar)
                || CharacterSet.decimalDigits.contains(scalar)
        }
        let tokens = expanded
            .lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { !$0.isEmpty }
        var candidates = Set(tokens)
        if !tokens.isEmpty {
            candidates.insert(tokens.joined(separator: "-"))
            candidates.insert(tokens.joined())
        }
        return candidates
    }

    private static func hasDuplicatePortShapes(_ shapes: [String]) -> Bool {
        Set(shapes).count != shapes.count
    }

    private static func hasCompletePortNames(_ names: [String], count: Int) -> Bool {
        names.count == count
            && !names.contains(where: \.isEmpty)
            && Set(names).count == names.count
    }

    private static func runtimeAssemblyEvidenceIdentityFields(
        _ evidence: RuntimeAssemblyExpectation
    ) -> [String] {
        [
            evidence.surfaceID,
            evidence.evidenceID,
            evidence.exportID,
            evidence.flowID,
            evidence.assemblyPlanKey,
            evidence.executionBackend,
            evidence.selectorDeletionLintID,
        ] + evidence.artifactRoles
            + evidence.gateIDs
            + evidence.requiredFeatureSet
            + evidence.consumedFeatureSet
            + evidence.unsupportedFeatureSet
    }

    static func runtimeContractCoverageKeys(
        _ expectations: [CapabilityExpectation]
    ) -> [String] {
        expectations.map(runtimeContractCoverageKey).sorted()
    }

    static func runtimeContractCoverageKeys(
        _ evidence: [RuntimeContractExpectation]
    ) -> [String] {
        evidence.map(runtimeContractCoverageKey).sorted()
    }

    static func runtimeRequiredCapabilityExpectations(
        for module: Module
    ) -> [CapabilityExpectation] {
        let runtimeSurfaceIDs = Set(
            module.surfaceExpectations
                .filter(\.requiresRuntimeContract)
                .map(\.surfaceID)
        )
        return module.capabilityExpectations.filter {
            runtimeSurfaceIDs.contains(runtimeContractSurfaceID(for: $0))
        }
    }

    static func runtimeContractCoverageKey(_ expectation: CapabilityExpectation) -> String {
        runtimeContractCoverageKey(
            surfaceID: runtimeContractSurfaceID(for: expectation),
            request: expectation.request,
            exportID: expectation.exportID,
            flowID: expectation.flowID,
            gateIDs: expectation.gateIDs
        )
    }

    static func runtimeContractCoverageKey(_ evidence: RuntimeContractExpectation) -> String {
        runtimeContractCoverageKey(
            surfaceID: evidence.surfaceID,
            request: evidence.request,
            exportID: evidence.exportID,
            flowID: evidence.flowID,
            gateIDs: evidence.gateIDs
        )
    }

    static func runtimeContractCoverageKey(
        surfaceID: String,
        request: SmeltCAMCapabilityRequest,
        exportID: String,
        flowID: String,
        gateIDs: [String]
    ) -> String {
        [
            surfaceID,
            request.name,
            exportID,
            flowID,
            gateIDs.sorted().joined(separator: ","),
        ].joined(separator: "|")
    }

    private static func runtimeContractSurfaceID(for expectation: CapabilityExpectation) -> String {
        if let surfaceID = expectation.surfaceID {
            return surfaceID
        }
        if expectation.request == .runText {
            return "run.text"
        }
        if expectation.request == .benchDecode {
            return "bench.text"
        }
        if expectation.request == .serveText {
            return "serve.text"
        }
        if expectation.request == .traceTextGenerate {
            return "trace.text"
        }
        if expectation.request == .bakeTextPromptPrefix {
            return "bake.prompt-prefix"
        }
        if expectation.request == .runAudio {
            return "run.audio-24khz"
        }
        if expectation.request == .benchAudio {
            return "bench.realtime"
        }
        if expectation.request == .serveAudio || expectation.request == .serveAudioStream {
            return "serve.audio-24khz"
        }
        if expectation.request == .traceTextSynthesize {
            return "trace.audio-24khz"
        }
        if expectation.request == .bakeVoiceDefaults {
            return "bake.voice-defaults"
        }
        return expectation.request.name
    }

    private enum MatrixError: Error, CustomStringConvertible {
        case unknownFixture(String)
        case missingRuntimeAssemblyRequest(String, String)
        case missingReleaseGate(String, String)
        case missingReleaseSurface(String, String)
        case missingReleaseExport(String, String)
        case missingReleaseFlow(String, String)
        case missingReleaseOutput(String, String, String)
        case missingReleaseRequirement(String, String)
        case missingReleaseGateContract(String, String)
        case invalidSelectorDeletionScan(String)

        var description: String {
            switch self {
            case .unknownFixture(let fixture):
                return "unknown CAM completion fixture: \(fixture)"
            case .missingRuntimeAssemblyRequest(let fixture, let surfaceID):
                return "\(fixture) runtime assembly surface '\(surfaceID)' has no CAM capability request"
            case .missingReleaseGate(let fixture, let gateID):
                return "\(fixture) release gate '\(gateID)' is not parsed from CAM"
            case .missingReleaseSurface(let fixture, let gateID):
                return "\(fixture) release gate '\(gateID)' has no release surface"
            case .missingReleaseExport(let fixture, let exportID):
                return "\(fixture) release export '\(exportID)' is not parsed from CAM"
            case .missingReleaseFlow(let fixture, let exportID):
                return "\(fixture) release export '\(exportID)' has no flow binding"
            case .missingReleaseOutput(let fixture, let gateID, let output):
                return "\(fixture) release gate '\(gateID)' references unknown output '\(output)'"
            case .missingReleaseRequirement(let fixture, let gateID):
                return "\(fixture) release gate '\(gateID)' has no requirement"
            case .missingReleaseGateContract(let fixture, let gateID):
                return "\(fixture) release gate '\(gateID)' has no projected contract"
            case .invalidSelectorDeletionScan(let detail):
                return "invalid selector deletion scan: \(detail)"
            }
        }
    }

    private static let twoTextRunRequest = SmeltCAMCapabilityRequest.exactTextToText(
        name: "run text with two required text inputs",
        requiredTextInputCount: 2,
        requiredInputNames: ["candidate", "context"],
        requiredAnyExportFacts: ["run.generate"]
    )

    private static let twoTextBenchRequest = SmeltCAMCapabilityRequest.exactTextToText(
        name: "bench text with two required text inputs",
        requiredTextInputCount: 2,
        requiredInputNames: ["candidate", "context"],
        requiredAnyExportFacts: ["run.generate"],
        requiredGateObservations: [SmeltCAMCapabilityRequest.firstTextOutputObservation()]
    )

    private static let twoTextServeRequest = SmeltCAMCapabilityRequest.exactTextToText(
        name: "serve text with two required text inputs",
        requiredTextInputCount: 2,
        requiredInputNames: ["candidate", "context"],
        requiredAnyExportFacts: ["run.generate"]
    )

    private static let twoTextTraceRequest = SmeltCAMCapabilityRequest.exactTextToText(
        name: "trace text with two required text inputs",
        requiredTextInputCount: 2,
        requiredInputNames: ["candidate", "context"],
        requiredAnyExportFacts: ["run.generate"]
    )

    private static let twoTextBakeRequest = SmeltCAMCapabilityRequest.exactTextToText(
        name: "bake prompt-prefix with two required text inputs",
        requiredTextInputCount: 2,
        requiredInputNames: ["candidate", "context"],
        requiredAnyExportFacts: ["bake.prompt-prefix"]
    )

    private static let tracePackageOpeningSubcommands = [
        "record",
        "replay",
        "suite",
        "verify",
    ]

    private static func commandSurfaceObligations(
        for surface: SurfaceExpectation
    ) -> [CommandSurfaceObligationExpectation] {
        guard surface.requiresRuntimeContract || surface.requiresRuntimeAssembly else {
            return []
        }
        if surface.surfaceID.hasPrefix("trace.") {
            return [
                commandSurfaceObligation(surface, command: "trace", subcommand: "record", runtimeEventSource: "case", deliveryMode: "package-opening"),
                commandSurfaceObligation(surface, command: "trace", subcommand: "verify", runtimeEventSource: "golden-case", deliveryMode: "package-opening"),
                commandSurfaceObligation(surface, command: "trace", subcommand: "replay", runtimeEventSource: "trace-case", deliveryMode: "package-opening"),
                commandSurfaceObligation(surface, command: "trace", subcommand: "suite", runtimeEventSource: "suite-case", deliveryMode: "package-opening"),
            ]
        }
        if surface.surfaceID.hasPrefix("linger.") {
            return [
                commandSurfaceObligation(
                    surface,
                    command: "linger-worker",
                    subcommand: "worker",
                    runtimeEventSource: "warm-worker",
                    deliveryMode: "warm-worker"
                ),
            ]
        }
        return []
    }

    private static func commandSurfaceObligation(
        _ surface: SurfaceExpectation,
        command: String,
        subcommand: String,
        runtimeEventSource: String,
        deliveryMode: String
    ) -> CommandSurfaceObligationExpectation {
        CommandSurfaceObligationExpectation(
            surfaceID: surface.surfaceID,
            command: command,
            subcommand: subcommand,
            runtimeEventSource: runtimeEventSource,
            deliveryMode: deliveryMode,
            counted: true,
            requiresRuntimeContract: surface.requiresRuntimeContract,
            requiresRuntimeAssembly: surface.requiresRuntimeAssembly,
            bridgeAbsentRequired: true
        )
    }

    private static func evidenceObligations(
        for module: Module,
        includeRuntimeAssemblyFeatureSets: Bool,
        requiring predicate: (SurfaceExpectation) -> Bool
    ) throws -> [EvidenceObligationExpectation] {
        try module.surfaceExpectations
            .filter(predicate)
            .flatMap {
                try evidenceObligations(
                    for: $0,
                    module: module,
                    includeRuntimeAssemblyFeatureSets: includeRuntimeAssemblyFeatureSets
                )
            }
    }

    private static func releaseSurfaceRefs(for module: Module) -> [String] {
        module.surfaceExpectations
            .filter(\.requiresReleaseEvidence)
            .map { "\(module.fixture):\($0.surfaceID)" }
            .sorted()
    }

    private static func evidenceObligations(
        for surface: SurfaceExpectation,
        module: Module,
        includeRuntimeAssemblyFeatureSets: Bool
    ) throws -> [EvidenceObligationExpectation] {
        let commandObligations = commandSurfaceObligations(for: surface)
        let featureContract = try includeRuntimeAssemblyFeatureSets
            ? runtimeAssemblyFeatureContract(for: surface, module: module)
            : nil
        let featureAdmission = try includeRuntimeAssemblyFeatureSets
            ? runtimeAssemblyFeatureAdmission(for: module)
            : nil
        let featureContractSchema = featureContract.map { _ in runtimeAssemblyFeatureContractSchema }
        let configuredGraphFeatureSet = featureContract?.configuredGraphFeatureSet ?? []
        let featureSet = featureContract?.featureSet ?? []
        let requiredObligationIDs = featureAdmission?.requiredObligationIDs ?? []
        let requiredObligationIDSet = Set(requiredObligationIDs)
        let consumedObligationIDs = Array(
            Set(module.consumedAdmissionObligationIDs).intersection(requiredObligationIDSet)
        ).sorted()
        let unsupportedObligationIDs = Array(
            requiredObligationIDSet.subtracting(consumedObligationIDs)
        ).sorted()
        if !commandObligations.isEmpty {
            return commandObligations.map { obligation in
                EvidenceObligationExpectation(
                    obligationKey: [
                        surface.surfaceID,
                        obligation.command,
                        obligation.subcommand,
                    ].joined(separator: ":"),
                    surfaceID: surface.surfaceID,
                    commandSurfaceObligationKey: [
                        surface.surfaceID,
                        obligation.command,
                        obligation.subcommand,
                    ].joined(separator: "|"),
                    command: obligation.command,
                    subcommand: obligation.subcommand,
                    runtimeEventSource: obligation.runtimeEventSource,
                    deliveryMode: obligation.deliveryMode,
                    counted: obligation.counted,
                    role: surface.kind.rawValue,
                    exportID: surface.exportID,
                    flowID: surface.flowID,
                    gateIDs: surface.gateIDs,
                    selectedInputNames: surface.selectedInputNames,
                    selectedInputs: surface.selectedInputs,
                    selectedOutputNames: surface.selectedOutputNames,
                    selectedOutputs: surface.selectedOutputs,
                    requiresReleaseEvidence: surface.requiresReleaseEvidence,
                    requiresRuntimeContract: surface.requiresRuntimeContract,
                    requiresRuntimeAssembly: surface.requiresRuntimeAssembly,
                    adapterBoundary: surface.adapterBoundary,
                    runtimeAssemblyFeatureContractSchema: featureContractSchema,
                    configuredGraphFeatureSet: configuredGraphFeatureSet,
                    requiredFeatureSet: featureSet,
                    consumedFeatureSet: featureSet,
                    unsupportedFeatureSet: [],
                    requiredObligationIDs: requiredObligationIDs,
                    consumedObligationIDs: consumedObligationIDs,
                    unsupportedObligationIDs: unsupportedObligationIDs
                )
            }
        }
        let commandParts = directCommandParts(for: surface)
        return [
            EvidenceObligationExpectation(
                obligationKey: surface.surfaceID,
                surfaceID: surface.surfaceID,
                commandSurfaceObligationKey: "",
                command: commandParts.command,
                subcommand: commandParts.subcommand,
                runtimeEventSource: commandParts.runtimeEventSource,
                deliveryMode: commandParts.deliveryMode,
                counted: true,
                role: surface.kind.rawValue,
                exportID: surface.exportID,
                flowID: surface.flowID,
                gateIDs: surface.gateIDs,
                selectedInputNames: surface.selectedInputNames,
                selectedInputs: surface.selectedInputs,
                selectedOutputNames: surface.selectedOutputNames,
                selectedOutputs: surface.selectedOutputs,
                requiresReleaseEvidence: surface.requiresReleaseEvidence,
                requiresRuntimeContract: surface.requiresRuntimeContract,
                requiresRuntimeAssembly: surface.requiresRuntimeAssembly,
                adapterBoundary: surface.adapterBoundary,
                runtimeAssemblyFeatureContractSchema: featureContractSchema,
                configuredGraphFeatureSet: configuredGraphFeatureSet,
                requiredFeatureSet: featureSet,
                consumedFeatureSet: featureSet,
                unsupportedFeatureSet: [],
                requiredObligationIDs: requiredObligationIDs,
                consumedObligationIDs: consumedObligationIDs,
                unsupportedObligationIDs: unsupportedObligationIDs
            ),
        ]
    }

    private static func runtimeAssemblyFeatureAdmission(
        for module: Module
    ) throws -> SmeltCAMFeatureAdmission {
        let ir = registryModuleIR(module.fixture)
        let descriptor = try SmeltCAMPackageDescriptor(from: ir)
        return SmeltCAMFeatureAdmission(
            descriptor: descriptor,
            consumedObligationIDs: module.consumedAdmissionObligationIDs
        )
    }

    private static func runtimeAssemblyFeatureContract(
        for surface: SurfaceExpectation,
        module: Module
    ) throws -> SmeltCAMPackageCapabilities.RuntimeAssemblyFeatureContract {
        guard let request = surface.request else {
            throw MatrixError.missingRuntimeAssemblyRequest(module.fixture, surface.surfaceID)
        }
        return try runtimeAssemblyFeatureContract(fixture: module.fixture, request: request)
    }

    private static func runtimeAssemblyFeatureContract(
        fixture: String,
        request: SmeltCAMCapabilityRequest
    ) throws -> SmeltCAMPackageCapabilities.RuntimeAssemblyFeatureContract {
        let ir = registryModuleIR(fixture)
        let capabilities = try SmeltCAMPackageCapabilities(
            descriptor: SmeltCAMPackageDescriptor(from: ir)
        )
        let decision = try capabilities.resolve(request)
        return try capabilities.runtimeAssemblyFeatureContract(for: decision)
    }

    private static func directCommandParts(
        for surface: SurfaceExpectation
    ) -> (command: String, subcommand: String, runtimeEventSource: String, deliveryMode: String) {
        let parts = surface.surfaceID.split(separator: ".", maxSplits: 1).map(String.init)
        let surfaceCommand = parts.first ?? surface.surfaceID
        let command = surfaceCommand == "bake" ? "create" : surfaceCommand
        let subcommand = parts.count == 2 ? parts[1] : surface.kind.rawValue
        let deliveryMode: String
        switch command {
        case "run":
            deliveryMode = "inline"
        case "bench":
            deliveryMode = "benchmark"
        case "serve":
            deliveryMode = "service"
        case "create":
            deliveryMode = "package-mutation"
        default:
            deliveryMode = surface.kind.rawValue
        }
        return (
            command: command,
            subcommand: subcommand,
            runtimeEventSource: "\(command)-\(subcommand)",
            deliveryMode: deliveryMode
        )
    }

    private static func textSurfaceExpectations(
        includeBake: Bool,
        exportGateIDs: [String] = ["startup"],
        releaseGateIDs: [String],
        buildAdapterBoundary: AdapterBoundary = .checkedProjectionProfile,
        commandAdapterBoundary: AdapterBoundary = .manifestPolicyBridge,
        lingerAdapterBoundary: AdapterBoundary? = nil,
        lingerRequiresRuntimeContract: Bool = true,
        lingerRequiresRuntimeAssembly: Bool = true,
        inventoryGateAdapterBoundary: AdapterBoundary? = nil,
        releaseGateAdapterBoundary: AdapterBoundary = .releaseBucketAdapter,
        releaseVerificationAdapterBoundary: AdapterBoundary = .targetVerifierBucket
    ) -> [SurfaceExpectation] {
        var surfaces: [SurfaceExpectation] = [
            buildSurface(adapterBoundary: buildAdapterBoundary),
            commandSurface(
                "run.text",
                request: .runText,
                selectedInputs: ["text[encoding=utf8]"],
                selectedOutputs: ["text[encoding=utf8]"],
                gateIDs: exportGateIDs,
                adapterBoundary: commandAdapterBoundary
            ),
            commandSurface(
                "bench.text",
                request: .benchDecode,
                selectedInputs: ["text[encoding=utf8]"],
                selectedOutputs: ["text[encoding=utf8]"],
                gateIDs: ["startup"],
                adapterBoundary: commandAdapterBoundary
            ),
            commandSurface(
                "serve.text",
                request: .serveText,
                selectedInputs: ["text[encoding=utf8]"],
                selectedOutputs: ["text[encoding=utf8]"],
                gateIDs: exportGateIDs,
                adapterBoundary: commandAdapterBoundary
            ),
            commandSurface(
                "trace.text",
                request: .traceTextGenerate,
                selectedInputs: ["text[encoding=utf8]"],
                selectedOutputs: ["text[encoding=utf8]"],
                gateIDs: exportGateIDs,
                adapterBoundary: commandAdapterBoundary
            ),
            commandSurface(
                "linger.text",
                request: .runText,
                selectedInputs: ["text[encoding=utf8]"],
                selectedOutputs: ["text[encoding=utf8]"],
                gateIDs: exportGateIDs,
                requiresRuntimeContract: lingerRequiresRuntimeContract,
                requiresRuntimeAssembly: lingerRequiresRuntimeAssembly,
                adapterBoundary: lingerAdapterBoundary ?? commandAdapterBoundary
            ),
            gateSurface(
                "gate.startup",
                gateIDs: ["startup"],
                requiresReleaseEvidence: true,
                adapterBoundary: releaseGateAdapterBoundary
            ),
            gateSurface(
                "gate.inventory",
                gateIDs: ["inventory"],
                requiresReleaseEvidence: false,
                adapterBoundary: inventoryGateAdapterBoundary
            ),
        ]
        if includeBake {
            surfaces.insert(
                commandSurface(
                    "bake.prompt-prefix",
                    request: .bakeTextPromptPrefix,
                    selectedInputs: ["text[encoding=utf8]"],
                    selectedOutputs: ["text[encoding=utf8]"],
                    gateIDs: exportGateIDs,
                    adapterBoundary: commandAdapterBoundary
                ),
                at: 4
            )
        }
        for gateID in releaseGateIDs where gateID != "startup" {
            surfaces.append(
                gateSurface(
                    "gate.\(gateID)",
                    gateIDs: [gateID],
                    requiresReleaseEvidence: true,
                    adapterBoundary: releaseGateAdapterBoundary
                )
            )
        }
        surfaces.append(
            releaseVerificationSurface(
                gateIDs: releaseGateIDs,
                adapterBoundary: releaseVerificationAdapterBoundary
            )
        )
        return surfaces
    }

    private static func twoTextSurfaceExpectations(
        releaseGateIDs: [String],
        buildAdapterBoundary: AdapterBoundary = .checkedProjectionProfile,
        commandAdapterBoundary: AdapterBoundary = .manifestPolicyBridge,
        lingerAdapterBoundary: AdapterBoundary? = nil,
        inventoryGateAdapterBoundary: AdapterBoundary? = nil,
        releaseGateAdapterBoundary: AdapterBoundary = .releaseBucketAdapter,
        releaseVerificationAdapterBoundary: AdapterBoundary = .targetVerifierBucket
    ) -> [SurfaceExpectation] {
        var surfaces: [SurfaceExpectation] = [
            buildSurface(adapterBoundary: buildAdapterBoundary),
            commandSurface(
                "run.text",
                request: twoTextRunRequest,
                selectedInputs: ["text[encoding=utf8]", "text[encoding=utf8]"],
                selectedOutputs: ["text[encoding=utf8]"],
                selectedInputNames: ["candidate", "context"],
                gateIDs: ["startup"],
                exportID: "review",
                flowID: "review",
                adapterBoundary: commandAdapterBoundary
            ),
            commandSurface(
                "bench.text",
                request: twoTextBenchRequest,
                selectedInputs: ["text[encoding=utf8]", "text[encoding=utf8]"],
                selectedOutputs: ["text[encoding=utf8]"],
                selectedInputNames: ["candidate", "context"],
                gateIDs: ["startup"],
                exportID: "review",
                flowID: "review",
                adapterBoundary: commandAdapterBoundary
            ),
            commandSurface(
                "serve.text",
                request: twoTextServeRequest,
                selectedInputs: ["text[encoding=utf8]", "text[encoding=utf8]"],
                selectedOutputs: ["text[encoding=utf8]"],
                selectedInputNames: ["candidate", "context"],
                gateIDs: ["startup"],
                exportID: "review",
                flowID: "review",
                adapterBoundary: commandAdapterBoundary
            ),
            commandSurface(
                "bake.prompt-prefix",
                request: twoTextBakeRequest,
                selectedInputs: ["text[encoding=utf8]", "text[encoding=utf8]"],
                selectedOutputs: ["text[encoding=utf8]"],
                selectedInputNames: ["candidate", "context"],
                gateIDs: ["startup"],
                exportID: "review",
                flowID: "review",
                adapterBoundary: commandAdapterBoundary
            ),
            commandSurface(
                "trace.text",
                request: twoTextTraceRequest,
                selectedInputs: ["text[encoding=utf8]", "text[encoding=utf8]"],
                selectedOutputs: ["text[encoding=utf8]"],
                selectedInputNames: ["candidate", "context"],
                gateIDs: ["startup"],
                exportID: "review",
                flowID: "review",
                adapterBoundary: commandAdapterBoundary
            ),
            commandSurface(
                "linger.text",
                request: twoTextRunRequest,
                selectedInputs: ["text[encoding=utf8]", "text[encoding=utf8]"],
                selectedOutputs: ["text[encoding=utf8]"],
                selectedInputNames: ["candidate", "context"],
                gateIDs: ["startup"],
                exportID: "review",
                flowID: "review",
                adapterBoundary: lingerAdapterBoundary ?? commandAdapterBoundary
            ),
            gateSurface(
                "gate.startup",
                exportID: "review",
                flowID: "review",
                gateIDs: ["startup"],
                requiresReleaseEvidence: true,
                adapterBoundary: releaseGateAdapterBoundary
            ),
            gateSurface(
                "gate.inventory",
                exportID: "review",
                flowID: "review",
                gateIDs: ["inventory"],
                requiresReleaseEvidence: false,
                adapterBoundary: inventoryGateAdapterBoundary
            ),
        ]
        for gateID in releaseGateIDs where gateID != "startup" {
            surfaces.append(
                gateSurface(
                    "gate.\(gateID)",
                    exportID: "review",
                    flowID: "review",
                    gateIDs: [gateID],
                    requiresReleaseEvidence: true,
                    adapterBoundary: releaseGateAdapterBoundary
                )
            )
        }
        surfaces.append(
            releaseVerificationSurface(
                exportID: "review",
                flowID: "review",
                gateIDs: releaseGateIDs,
                adapterBoundary: releaseVerificationAdapterBoundary
            )
        )
        return surfaces
    }

    private static func textToAudio24KSurfaceExpectations(
        commandAdapterBoundary: AdapterBoundary = .commandAdapter,
        traceRequiresRuntimeContract: Bool = true,
        traceRequiresRuntimeAssembly: Bool = true,
        traceAdapterBoundary: AdapterBoundary = .commandAdapter,
        optionalInputAdapterBoundary: AdapterBoundary = .manifestPolicyBridge,
        buildAdapterBoundary: AdapterBoundary = .checkedProjectionProfile,
        releaseCommandAdapterBoundary: AdapterBoundary = .commandAdapter,
        releaseGateAdapterBoundary: AdapterBoundary = .releaseBucketAdapter,
        releaseVerificationAdapterBoundary: AdapterBoundary = .targetVerifierBucket
    ) -> [SurfaceExpectation] {
        [
            buildSurface(adapterBoundary: buildAdapterBoundary),
            commandSurface(
                "run.audio-24khz",
                request: .runAudio,
                selectedInputs: ["text[encoding=utf8]"],
                selectedOutputs: ["pcm[dtype=f32,rate=24khz]"],
                selectedInputNames: ["text"],
                gateIDs: ["startup"],
                exportID: "synth",
                flowID: "synth",
                adapterBoundary: commandAdapterBoundary
            ),
            optionalInputSurface(
                "optional.speaker",
                exportID: "synth",
                flowID: "synth",
                gateIDs: ["startup"],
                adapterBoundary: optionalInputAdapterBoundary
            ),
            commandSurface(
                "serve.audio-24khz",
                request: .serveAudio,
                selectedInputs: ["text[encoding=utf8]"],
                selectedOutputs: ["pcm[dtype=f32,rate=24khz]"],
                selectedInputNames: ["text"],
                gateIDs: ["startup"],
                exportID: "synth",
                flowID: "synth",
                adapterBoundary: commandAdapterBoundary
            ),
            commandSurface(
                "linger.audio-24khz",
                request: .runAudio,
                selectedInputs: ["text[encoding=utf8]"],
                selectedOutputs: ["pcm[dtype=f32,rate=24khz]"],
                selectedInputNames: ["text"],
                gateIDs: ["startup"],
                exportID: "synth",
                flowID: "synth",
                adapterBoundary: commandAdapterBoundary
            ),
            commandSurface(
                "trace.audio-24khz",
                request: .traceTextSynthesize,
                selectedInputs: ["text[encoding=utf8]"],
                selectedOutputs: ["pcm[dtype=f32,rate=24khz]"],
                selectedInputNames: ["text"],
                gateIDs: ["startup"],
                exportID: "synth",
                flowID: "synth",
                requiresRuntimeContract: traceRequiresRuntimeContract,
                requiresRuntimeAssembly: traceRequiresRuntimeAssembly,
                adapterBoundary: traceAdapterBoundary
            ),
            gateSurface(
                "gate.startup-audio",
                exportID: "synth",
                flowID: "synth",
                gateIDs: ["startup"],
                requiresReleaseEvidence: true,
                adapterBoundary: releaseGateAdapterBoundary
            ),
            gateSurface(
                "gate.audio-contract",
                exportID: "synth",
                flowID: "synth",
                gateIDs: ["audio_contract"],
                requiresReleaseEvidence: true,
                adapterBoundary: releaseGateAdapterBoundary
            ),
            correctnessSurface(
                "correctness.stream-parity",
                exportID: "synth",
                flowID: "synth",
                gateIDs: ["streaming_parity"],
                adapterBoundary: releaseVerificationAdapterBoundary
            ),
            releaseVerificationSurface(
                exportID: "synth",
                flowID: "synth",
                gateIDs: ["startup", "audio_contract", "streaming_parity"],
                adapterBoundary: releaseVerificationAdapterBoundary
            ),
        ]
    }

    private static func buildSurface(
        adapterBoundary: AdapterBoundary = .checkedProjectionProfile
    ) -> SurfaceExpectation {
        SurfaceExpectation(
            surfaceID: "build",
            kind: .build,
            request: nil,
            exportID: nil,
            flowID: nil,
            selectedInputs: [],
            selectedOutputs: [],
            selectedInputNames: [],
            selectedOutputNames: [],
            gateIDs: [],
            requiresRuntimeContract: false,
            requiresReleaseEvidence: false,
            requiresRuntimeAssembly: false,
            adapterBoundary: adapterBoundary
        )
    }

    private static func commandSurface(
        _ surfaceID: String,
        request: SmeltCAMCapabilityRequest,
        selectedInputs: [String],
        selectedOutputs: [String],
        selectedInputNames: [String]? = nil,
        selectedOutputNames: [String]? = nil,
        gateIDs: [String],
        exportID: String = "generate",
        flowID: String = "generate",
        requiresReleaseEvidence: Bool = false,
        requiresRuntimeContract: Bool = true,
        requiresRuntimeAssembly: Bool = true,
        adapterBoundary: AdapterBoundary = .manifestPolicyBridge
    ) -> SurfaceExpectation {
        SurfaceExpectation(
            surfaceID: surfaceID,
            kind: .command,
            request: request,
            exportID: exportID,
            flowID: flowID,
            selectedInputs: selectedInputs,
            selectedOutputs: selectedOutputs,
            selectedInputNames: selectedInputNames ?? (selectedInputs.count == 1 ? ["prompt"] : []),
            selectedOutputNames: selectedOutputNames ?? (selectedOutputs.first?.hasPrefix("pcm[") == true ? ["audio"] : ["text"]),
            gateIDs: gateIDs,
            requiresRuntimeContract: requiresRuntimeContract,
            requiresReleaseEvidence: requiresReleaseEvidence,
            requiresRuntimeAssembly: requiresRuntimeAssembly,
            adapterBoundary: adapterBoundary
        )
    }

    private static func gateSurface(
        _ surfaceID: String,
        exportID: String = "generate",
        flowID: String = "generate",
        gateIDs: [String],
        requiresReleaseEvidence: Bool,
        adapterBoundary: AdapterBoundary? = nil
    ) -> SurfaceExpectation {
        SurfaceExpectation(
            surfaceID: surfaceID,
            kind: .gate,
            request: nil,
            exportID: exportID,
            flowID: flowID,
            selectedInputs: [],
            selectedOutputs: [],
            selectedInputNames: [],
            selectedOutputNames: [],
            gateIDs: gateIDs,
            requiresRuntimeContract: false,
            requiresReleaseEvidence: requiresReleaseEvidence,
            requiresRuntimeAssembly: false,
            adapterBoundary: adapterBoundary
                ?? (requiresReleaseEvidence ? .releaseBucketAdapter : .checkedProjectionProfile)
        )
    }

    private static func optionalInputSurface(
        _ surfaceID: String,
        exportID: String,
        flowID: String,
        gateIDs: [String],
        adapterBoundary: AdapterBoundary = .manifestPolicyBridge
    ) -> SurfaceExpectation {
        SurfaceExpectation(
            surfaceID: surfaceID,
            kind: .optionalInputCoverage,
            request: nil,
            exportID: exportID,
            flowID: flowID,
            selectedInputs: [],
            selectedOutputs: [],
            selectedInputNames: [],
            selectedOutputNames: [],
            gateIDs: gateIDs,
            requiresRuntimeContract: false,
            requiresReleaseEvidence: false,
            requiresRuntimeAssembly: false,
            adapterBoundary: adapterBoundary
        )
    }

    private static func correctnessSurface(
        _ surfaceID: String,
        exportID: String,
        flowID: String,
        gateIDs: [String],
        adapterBoundary: AdapterBoundary = .targetVerifierBucket
    ) -> SurfaceExpectation {
        SurfaceExpectation(
            surfaceID: surfaceID,
            kind: .correctness,
            request: nil,
            exportID: exportID,
            flowID: flowID,
            selectedInputs: [],
            selectedOutputs: [],
            selectedInputNames: [],
            selectedOutputNames: [],
            gateIDs: gateIDs,
            requiresRuntimeContract: false,
            requiresReleaseEvidence: true,
            requiresRuntimeAssembly: false,
            adapterBoundary: adapterBoundary
        )
    }

    private static func releaseVerificationSurface(
        exportID: String = "generate",
        flowID: String = "generate",
        gateIDs: [String],
        adapterBoundary: AdapterBoundary = .targetVerifierBucket
    ) -> SurfaceExpectation {
        SurfaceExpectation(
            surfaceID: "release.verify",
            kind: .releaseVerification,
            request: nil,
            exportID: exportID,
            flowID: flowID,
            selectedInputs: [],
            selectedOutputs: [],
            selectedInputNames: [],
            selectedOutputNames: [],
            gateIDs: gateIDs,
            requiresRuntimeContract: false,
            requiresReleaseEvidence: true,
            requiresRuntimeAssembly: false,
            adapterBoundary: adapterBoundary
        )
    }

    private static func closedSurface(
        _ surfaceID: String,
        surface: String,
        reason: String,
        gateIDs: [String]
    ) -> ClosedSurfaceExpectation {
        ClosedSurfaceExpectation(
            surfaceID: surfaceID,
            surface: surface,
            reason: reason,
            requiredGateIDs: gateIDs
        )
    }

    private static func textClosedSurfaces(
        includeBake: Bool,
        exportGateIDs: [String] = ["startup"],
        releaseGateIDs: [String],
        includeRuntimeSurfaceBlockers: Bool = true,
        includeReleaseEvidenceBlockers: Bool = true
    ) -> [ClosedSurfaceExpectation] {
        var blockers: [ClosedSurfaceExpectation] = []
        if includeRuntimeSurfaceBlockers {
            blockers += [
                closedSurface(
                    "run.text",
                    surface: "run text",
                    reason: "capability route is not yet cut over for text run",
                    gateIDs: exportGateIDs
                ),
                closedSurface(
                    "bench.text",
                    surface: "bench text",
                    reason: "capability route is not yet cut over for text decode benchmark",
                    gateIDs: ["startup"]
                ),
                closedSurface(
                    "serve.text",
                    surface: "serve text",
                    reason: "capability route is not yet cut over for text generation serve",
                    gateIDs: exportGateIDs
                ),
            ]
        }
        if includeBake && includeRuntimeSurfaceBlockers {
            blockers.append(
                closedSurface(
                    "bake.prompt-prefix",
                    surface: "bake prompt-prefix",
                    reason: "capability route is not yet cut over for prompt-prefix bake",
                    gateIDs: exportGateIDs
                )
            )
        }
        if includeRuntimeSurfaceBlockers {
            blockers.append(
                closedSurface(
                    "trace.text",
                    surface: "trace text",
                    reason: "capability route is not yet cut over for text generation trace",
                    gateIDs: exportGateIDs
                )
            )
        }
        blockers.append(
            closedSurface(
                "linger.text",
                surface: "linger text",
                reason: "capability route is not yet cut over for text generation linger worker",
                gateIDs: exportGateIDs
            )
        )
        if includeReleaseEvidenceBlockers {
            blockers.append(
                closedSurface(
                    "gate.startup",
                    surface: "startup gate",
                    reason: "startup gate evidence still crosses release bucket adapter",
                    gateIDs: ["startup"]
                )
            )
            blockers.append(
                closedSurface(
                    "gate.inventory",
                    surface: "inventory gate",
                    reason: "inventory gate still crosses checked projection profile",
                    gateIDs: ["inventory"]
                )
            )
            for gateID in releaseGateIDs where gateID != "startup" {
                blockers.append(
                    closedSurface(
                        "gate.\(gateID)",
                        surface: "\(gateID) gate",
                        reason: "\(gateID) gate evidence still crosses release bucket adapter",
                        gateIDs: [gateID]
                    )
                )
            }
            blockers.append(
                closedSurface(
                    "release.verify",
                    surface: "release verify",
                    reason: "release verification still crosses target verifier bucket",
                    gateIDs: releaseGateIDs
                )
            )
        } else {
            blockers.append(
                closedSurface(
                    "gate.inventory",
                    surface: "inventory gate",
                    reason: "inventory gate still crosses checked projection profile",
                    gateIDs: ["inventory"]
                )
            )
        }
        return blockers
    }

    private static func textReleaseGateEvidence(
        packageFiles: [String],
        camSemanticSHA256: String,
        exportABISHA256: String,
        camDescriptorSHA256: String,
        descriptorGraphSignatureSHA256: String,
        packageProjectionID: String,
        projectedPackageSpecSHA256: String,
        buildEvidenceSHA256: String,
        packagePayloadSHA256: String,
        elapsedBound: String,
        includePrefillDecode: Bool,
        includeStorage: Bool = false,
        includeProjectionBias: Bool = false,
        includeLongContext: Bool = false,
        includePackageIdentity: Bool = false,
        packageIdentityBound: String? = nil,
        prefillBound: String = "256",
        exportID: String = "generate",
        flowID: String = "generate",
        selectedCapability: String = "run.generate",
        outputName: String = "text"
    ) -> [ReleaseGateExpectation] {
        let acceptedEventID = "flow.accepted:\(flowID)"
        let outputEventID = "emit:\(flowID).\(outputName)"
        func row(
            surfaceID: String,
            evidenceID: String,
            gateID: String,
            gateTraceLabels: [String],
            metric: String,
            metricPath: String,
            comparator: String,
            bound: String,
            unit: String?,
            fromEventID: String,
            toEventID: String,
            commandStamp: String
        ) -> ReleaseGateExpectation {
            let measurementSHA256 = releaseEvidenceSHA256(
                "smelt.module.release.measurement.v1",
                [
                    surfaceID,
                    evidenceID,
                    gateID,
                    gateTraceLabels.joined(separator: ","),
                    metric,
                    metricPath,
                    comparator,
                    bound,
                    unit ?? "",
                    fromEventID,
                    toEventID,
                    "process-cold",
                    "2026-06-30T00:00:00Z",
                    "process-cold",
                    commandStamp,
                    packagePayloadSHA256,
                ]
            )
            let provenanceSHA256 = releaseEvidenceSHA256(
                "smelt.module.release.provenance.v1",
                [
                    surfaceID,
                    evidenceID,
                    camSemanticSHA256,
                    exportABISHA256,
                    camDescriptorSHA256,
                    descriptorGraphSignatureSHA256,
                    packageProjectionID,
                    String(1),
                    projectedPackageSpecSHA256,
                    packageFiles.joined(separator: ","),
                    buildEvidenceSHA256,
                    packagePayloadSHA256,
                    commandStamp,
                ]
            )
            return ReleaseGateExpectation(
                surfaceID: surfaceID,
                evidenceID: evidenceID,
                request: nil,
                exportID: exportID,
                flowID: flowID,
                selectedCapability: selectedCapability,
                gateID: gateID,
                gateTraceLabels: gateTraceLabels,
                metric: metric,
                metricPath: metricPath,
                comparator: comparator,
                bound: bound,
                unit: unit,
                fromEventID: fromEventID,
                toEventID: toEventID,
                processMode: "process-cold",
                processStartTimestamp: "2026-06-30T00:00:00Z",
                cacheState: "process-cold",
                camSemanticSHA256: camSemanticSHA256,
                exportABISHA256: exportABISHA256,
                camDescriptorSHA256: camDescriptorSHA256,
                descriptorGraphSignatureSHA256: descriptorGraphSignatureSHA256,
                packageProjectionID: packageProjectionID,
                packageProjectionVersion: 1,
                projectedPackageSpecSHA256: projectedPackageSpecSHA256,
                packageFiles: packageFiles,
                buildEvidenceSHA256: buildEvidenceSHA256,
                packagePayloadSHA256: packagePayloadSHA256,
                commandStamp: commandStamp,
                measurementSHA256: measurementSHA256,
                provenanceSHA256: provenanceSHA256
            )
        }
        var rows = [
            row(
                surfaceID: "gate.startup",
                evidenceID: "release-startup-contract",
                gateID: "startup",
                gateTraceLabels: [acceptedEventID, outputEventID],
                metric: "elapsed",
                metricPath: "elapsed:\(acceptedEventID)->\(outputEventID)",
                comparator: "<=",
                bound: elapsedBound,
                unit: "ms",
                fromEventID: acceptedEventID,
                toEventID: outputEventID,
                commandStamp: "cam-release-startup"
            ),
        ]
        if includePrefillDecode {
            rows.append(
                row(
                    surfaceID: "gate.prefill",
                    evidenceID: "release-prefill-contract",
                    gateID: "prefill",
                    gateTraceLabels: ["gate.prefill"],
                    metric: "prefill-batch",
                    metricPath: "scalar-metric:gate.prefill:prefill-batch",
                    comparator: "<=",
                    bound: prefillBound,
                    unit: nil,
                    fromEventID: "",
                    toEventID: "",
                    commandStamp: "cam-release-prefill"
                )
            )
            rows.append(
                row(
                    surfaceID: "gate.decode",
                    evidenceID: "release-decode-contract",
                    gateID: "decode",
                    gateTraceLabels: ["gate.decode"],
                    metric: "decode-output.tokens",
                    metricPath: "scalar-metric:gate.decode:decode-output.tokens",
                    comparator: ">=",
                    bound: "1",
                    unit: nil,
                    fromEventID: "",
                    toEventID: "",
                    commandStamp: "cam-release-decode"
                )
            )
        }
        if includeStorage {
            rows.append(
                row(
                    surfaceID: "gate.storage",
                    evidenceID: "release-storage-contract",
                    gateID: "storage",
                    gateTraceLabels: ["gate.storage"],
                    metric: "evidence",
                    metricPath: "evidence:gate.storage:evidence",
                    comparator: "present",
                    bound: "2",
                    unit: nil,
                    fromEventID: "",
                    toEventID: "",
                    commandStamp: "cam-release-storage"
                )
            )
        }
        if includeProjectionBias {
            rows.append(
                row(
                    surfaceID: "gate.projection_bias",
                    evidenceID: "release-projection-bias-contract",
                    gateID: "projection_bias",
                    gateTraceLabels: ["gate.projection_bias"],
                    metric: "evidence",
                    metricPath: "evidence:gate.projection_bias:evidence",
                    comparator: "present",
                    bound: "3",
                    unit: nil,
                    fromEventID: "",
                    toEventID: "",
                    commandStamp: "cam-release-projection-bias"
                )
            )
        }
        if includeLongContext {
            rows.append(
                row(
                    surfaceID: "gate.long_context",
                    evidenceID: "release-long-context-contract",
                    gateID: "long_context",
                    gateTraceLabels: ["gate.long_context"],
                    metric: "static-seq-capacity",
                    metricPath: "scalar-metric:gate.long_context:static-seq-capacity",
                    comparator: ">=",
                    bound: "131072",
                    unit: nil,
                    fromEventID: "",
                    toEventID: "",
                    commandStamp: "cam-release-long-context"
                )
            )
        }
        if includePackageIdentity {
            rows.append(
                row(
                    surfaceID: "gate.package_identity",
                    evidenceID: "release-package-identity-contract",
                    gateID: "package_identity",
                    gateTraceLabels: ["gate.package_identity"],
                    metric: "package-files",
                    metricPath: "inventory:gate.package_identity:package-files",
                    comparator: "include",
                    bound: packageIdentityBound ?? packageFiles.joined(separator: ","),
                    unit: nil,
                    fromEventID: "",
                    toEventID: "",
                    commandStamp: "cam-release-package-identity"
                )
            )
        }
        rows.append(
            row(
                surfaceID: "release.verify",
                evidenceID: "release-verify-startup-contract",
                gateID: "startup",
                gateTraceLabels: [acceptedEventID, outputEventID],
                metric: "elapsed",
                metricPath: "elapsed:\(acceptedEventID)->\(outputEventID)",
                comparator: "<=",
                bound: elapsedBound,
                unit: "ms",
                fromEventID: acceptedEventID,
                toEventID: outputEventID,
                commandStamp: "cam-release-verify"
            )
        )
        if includePrefillDecode {
            rows.append(
                row(
                    surfaceID: "release.verify",
                    evidenceID: "release-verify-prefill-contract",
                    gateID: "prefill",
                    gateTraceLabels: ["gate.prefill"],
                    metric: "prefill-batch",
                    metricPath: "scalar-metric:gate.prefill:prefill-batch",
                    comparator: "<=",
                    bound: prefillBound,
                    unit: nil,
                    fromEventID: "",
                    toEventID: "",
                    commandStamp: "cam-release-verify-prefill"
                )
            )
            rows.append(
                row(
                    surfaceID: "release.verify",
                    evidenceID: "release-verify-decode-contract",
                    gateID: "decode",
                    gateTraceLabels: ["gate.decode"],
                    metric: "decode-output.tokens",
                    metricPath: "scalar-metric:gate.decode:decode-output.tokens",
                    comparator: ">=",
                    bound: "1",
                    unit: nil,
                    fromEventID: "",
                    toEventID: "",
                    commandStamp: "cam-release-verify-decode"
                )
            )
        }
        if includeStorage {
            rows.append(
                row(
                    surfaceID: "release.verify",
                    evidenceID: "release-verify-storage-contract",
                    gateID: "storage",
                    gateTraceLabels: ["gate.storage"],
                    metric: "evidence",
                    metricPath: "evidence:gate.storage:evidence",
                    comparator: "present",
                    bound: "2",
                    unit: nil,
                    fromEventID: "",
                    toEventID: "",
                    commandStamp: "cam-release-verify-storage"
                )
            )
        }
        if includeProjectionBias {
            rows.append(
                row(
                    surfaceID: "release.verify",
                    evidenceID: "release-verify-projection-bias-contract",
                    gateID: "projection_bias",
                    gateTraceLabels: ["gate.projection_bias"],
                    metric: "evidence",
                    metricPath: "evidence:gate.projection_bias:evidence",
                    comparator: "present",
                    bound: "3",
                    unit: nil,
                    fromEventID: "",
                    toEventID: "",
                    commandStamp: "cam-release-verify-projection-bias"
                )
            )
        }
        if includeLongContext {
            rows.append(
                row(
                    surfaceID: "release.verify",
                    evidenceID: "release-verify-long-context-contract",
                    gateID: "long_context",
                    gateTraceLabels: ["gate.long_context"],
                    metric: "static-seq-capacity",
                    metricPath: "scalar-metric:gate.long_context:static-seq-capacity",
                    comparator: ">=",
                    bound: "131072",
                    unit: nil,
                    fromEventID: "",
                    toEventID: "",
                    commandStamp: "cam-release-verify-long-context"
                )
            )
        }
        if includePackageIdentity {
            rows.append(
                row(
                    surfaceID: "release.verify",
                    evidenceID: "release-verify-package-identity-contract",
                    gateID: "package_identity",
                    gateTraceLabels: ["gate.package_identity"],
                    metric: "package-files",
                    metricPath: "inventory:gate.package_identity:package-files",
                    comparator: "include",
                    bound: packageIdentityBound ?? packageFiles.joined(separator: ","),
                    unit: nil,
                    fromEventID: "",
                    toEventID: "",
                    commandStamp: "cam-release-verify-package-identity"
                )
            )
        }
        return rows
    }

    private static func audio24KReleaseGateEvidence(
        packageFiles: [String],
        camSemanticSHA256: String,
        exportABISHA256: String,
        camDescriptorSHA256: String,
        descriptorGraphSignatureSHA256: String,
        packageProjectionID: String,
        projectedPackageSpecSHA256: String,
        buildEvidenceSHA256: String,
        packagePayloadSHA256: String,
        exportID: String = "synth",
        flowID: String = "synth"
    ) -> [ReleaseGateExpectation] {
        let acceptedEventID = "flow.accepted:\(flowID)"
        let outputEventID = "emit:\(flowID).audio"
        let packageFileInventory = [
            "manifest.json",
            "weights.bin",
            "model.metallib",
            "trunk",
            "trunk-mtp",
            "vocab.json",
            "merges.txt",
            "tokenizer_config.json",
            "config.json",
            "module.json",
        ].joined(separator: ",")
        func row(
            surfaceID: String,
            evidenceID: String,
            request: SmeltCAMCapabilityRequest? = nil,
            selectedCapability: String,
            gateID: String,
            gateTraceLabels: [String],
            metric: String,
            metricPath: String,
            comparator: String,
            bound: String,
            unit: String?,
            fromEventID: String,
            toEventID: String,
            commandStamp: String
        ) -> ReleaseGateExpectation {
            let measurementSHA256 = releaseEvidenceSHA256(
                "smelt.module.release.measurement.v1",
                [
                    surfaceID,
                    evidenceID,
                    gateID,
                    gateTraceLabels.joined(separator: ","),
                    metric,
                    metricPath,
                    comparator,
                    bound,
                    unit ?? "",
                    fromEventID,
                    toEventID,
                    "process-cold",
                    "2026-06-30T00:00:00Z",
                    "process-cold",
                    commandStamp,
                    packagePayloadSHA256,
                ]
            )
            let provenanceSHA256 = releaseEvidenceSHA256(
                "smelt.module.release.provenance.v1",
                [
                    surfaceID,
                    evidenceID,
                    camSemanticSHA256,
                    exportABISHA256,
                    camDescriptorSHA256,
                    descriptorGraphSignatureSHA256,
                    packageProjectionID,
                    String(1),
                    projectedPackageSpecSHA256,
                    packageFiles.joined(separator: ","),
                    buildEvidenceSHA256,
                    packagePayloadSHA256,
                    commandStamp,
                ]
            )
            return ReleaseGateExpectation(
                surfaceID: surfaceID,
                evidenceID: evidenceID,
                request: request,
                exportID: exportID,
                flowID: flowID,
                selectedCapability: selectedCapability,
                gateID: gateID,
                gateTraceLabels: gateTraceLabels,
                metric: metric,
                metricPath: metricPath,
                comparator: comparator,
                bound: bound,
                unit: unit,
                fromEventID: fromEventID,
                toEventID: toEventID,
                processMode: "process-cold",
                processStartTimestamp: "2026-06-30T00:00:00Z",
                cacheState: "process-cold",
                camSemanticSHA256: camSemanticSHA256,
                exportABISHA256: exportABISHA256,
                camDescriptorSHA256: camDescriptorSHA256,
                descriptorGraphSignatureSHA256: descriptorGraphSignatureSHA256,
                packageProjectionID: packageProjectionID,
                packageProjectionVersion: 1,
                projectedPackageSpecSHA256: projectedPackageSpecSHA256,
                packageFiles: packageFiles,
                buildEvidenceSHA256: buildEvidenceSHA256,
                packagePayloadSHA256: packagePayloadSHA256,
                commandStamp: commandStamp,
                measurementSHA256: measurementSHA256,
                provenanceSHA256: provenanceSHA256
            )
        }
        func startupRow(
            surfaceID: String,
            evidenceID: String,
            commandStamp: String,
            request: SmeltCAMCapabilityRequest? = nil,
            selectedCapability: String = "run.stream"
        ) -> ReleaseGateExpectation {
            row(
                surfaceID: surfaceID,
                evidenceID: evidenceID,
                request: request,
                selectedCapability: selectedCapability,
                gateID: "startup",
                gateTraceLabels: [acceptedEventID, outputEventID],
                metric: "elapsed",
                metricPath: "elapsed:\(acceptedEventID)->\(outputEventID)",
                comparator: "<=",
                bound: "400",
                unit: "ms",
                fromEventID: acceptedEventID,
                toEventID: outputEventID,
                commandStamp: commandStamp
            )
        }
        func inventoryRow(surfaceID: String, evidenceID: String, commandStamp: String) -> ReleaseGateExpectation {
            row(
                surfaceID: surfaceID,
                evidenceID: evidenceID,
                selectedCapability: "run.synthesize",
                gateID: "audio_contract",
                gateTraceLabels: ["gate.audio_contract"],
                metric: "package-files",
                metricPath: "inventory:gate.audio_contract:package-files",
                comparator: "include",
                bound: packageFileInventory,
                unit: nil,
                fromEventID: "",
                toEventID: "",
                commandStamp: commandStamp
            )
        }
        func evidenceRow(
            surfaceID: String,
            evidenceID: String,
            selectedCapability: String,
            gateID: String,
            commandStamp: String
        ) -> ReleaseGateExpectation {
            row(
                surfaceID: surfaceID,
                evidenceID: evidenceID,
                selectedCapability: selectedCapability,
                gateID: gateID,
                gateTraceLabels: ["gate.\(gateID)"],
                metric: "evidence",
                metricPath: "evidence:gate.\(gateID):evidence",
                comparator: "present",
                bound: "1",
                unit: nil,
                fromEventID: "",
                toEventID: "",
                commandStamp: commandStamp
            )
        }
        return [
            startupRow(
                surfaceID: "gate.startup-audio",
                evidenceID: "release-startup-contract",
                commandStamp: "cam-release-startup"
            ),
            inventoryRow(
                surfaceID: "gate.audio-contract",
                evidenceID: "release-audio-contract",
                commandStamp: "cam-release-audio-contract"
            ),
            evidenceRow(
                surfaceID: "correctness.stream-parity",
                evidenceID: "release-stream-parity",
                selectedCapability: "run.stream",
                gateID: "streaming_parity",
                commandStamp: "cam-release-stream-parity"
            ),
            startupRow(
                surfaceID: "release.verify",
                evidenceID: "release-verify-startup-contract",
                commandStamp: "cam-release-verify-startup"
            ),
            inventoryRow(
                surfaceID: "release.verify",
                evidenceID: "release-verify-audio-contract",
                commandStamp: "cam-release-verify-audio-contract"
            ),
            evidenceRow(
                surfaceID: "release.verify",
                evidenceID: "release-verify-stream-parity",
                selectedCapability: "run.stream",
                gateID: "streaming_parity",
                commandStamp: "cam-release-verify-stream-parity"
            ),
        ]
    }

    private static func releaseEvidenceSHA256(_ domain: String, _ fields: [String]) -> String {
        let payload = ([domain] + fields).joined(separator: "\n")
        return SHA256.hash(data: Data(payload.utf8))
            .map { String(format: "%02x", $0) }
            .joined()
    }

    private static func twoTextClosedSurfaces(
        releaseGateIDs: [String]
    ) -> [ClosedSurfaceExpectation] {
        [
            closedSurface(
                "run.text",
                surface: "run two-text review",
                reason: "runtime still crosses manifest-policy bridge for two-text review",
                gateIDs: ["startup"]
            ),
            closedSurface(
                "bench.text",
                surface: "bench two-text review",
                reason: "benchmark still crosses manifest-policy bridge for two-text review",
                gateIDs: ["startup"]
            ),
            closedSurface(
                "serve.text",
                surface: "serve two-text review",
                reason: "serve still crosses manifest-policy bridge for two-text review",
                gateIDs: ["startup"]
            ),
            closedSurface(
                "bake.prompt-prefix",
                surface: "bake two-text prompt-prefix",
                reason: "prompt-prefix bake still crosses manifest-policy bridge for two-text review",
                gateIDs: ["startup"]
            ),
            closedSurface(
                "trace.text",
                surface: "trace two-text review",
                reason: "trace still crosses manifest-policy bridge for two-text review",
                gateIDs: ["startup"]
            ),
            closedSurface(
                "linger.text",
                surface: "linger two-text review",
                reason: "linger worker still crosses manifest-policy bridge for two-text review",
                gateIDs: ["startup"]
            ),
            closedSurface(
                "gate.startup",
                surface: "startup gate",
                reason: "startup gate evidence still crosses release bucket adapter",
                gateIDs: ["startup"]
            ),
            closedSurface(
                "gate.inventory",
                surface: "inventory gate",
                reason: "inventory gate still crosses checked projection profile",
                gateIDs: ["inventory"]
            ),
            closedSurface(
                "release.verify",
                surface: "release verify",
                reason: "release verification still crosses target verifier bucket",
                gateIDs: releaseGateIDs
            ),
        ]
    }

    private static func textToAudio24KClosedSurfaces(
        includeRuntimeCommandBlockers: Bool = true,
        includeTraceCommandBlocker: Bool = false,
        includeReleaseEvidenceBlockers: Bool = true,
        includeOptionalInputBlocker: Bool = true
    ) -> [ClosedSurfaceExpectation] {
        var blockers: [ClosedSurfaceExpectation] = []
        if includeRuntimeCommandBlockers {
            blockers += [
                closedSurface(
                    "run.audio-24khz",
                    surface: "run 24khz audio",
                    reason: "runtime has CAM text-to-PCM construction but module is not promoted to native command evidence yet",
                    gateIDs: ["startup"]
                ),
                closedSurface(
                    "serve.audio-24khz",
                    surface: "serve 24khz audio",
                    reason: "serve has CAM text-to-PCM construction but module is not promoted to native command evidence yet",
                    gateIDs: ["startup"]
                ),
                closedSurface(
                    "linger.audio-24khz",
                    surface: "linger 24khz audio",
                    reason: "linger worker has CAM text-to-PCM construction but module is not promoted to native command evidence yet",
                    gateIDs: ["startup"]
                ),
                closedSurface(
                    "bake.voice-defaults",
                    surface: "bake voice defaults",
                    reason: "voice-default bake has CAM text-to-PCM construction but module is not promoted to native command evidence yet",
                    gateIDs: ["startup"]
                ),
                closedSurface(
                    "trace.audio-24khz",
                    surface: "trace 24khz audio",
                    reason: "trace has CAM text-to-PCM construction but module is not promoted to native command evidence yet",
                    gateIDs: ["startup"]
                ),
            ]
        }
        if includeTraceCommandBlocker && !includeRuntimeCommandBlockers {
            blockers.append(
                closedSurface(
                    "trace.audio-24khz",
                    surface: "trace 24khz audio",
                    reason: "trace can still take static/event-only paths and is not promoted to runtime construction evidence yet",
                    gateIDs: ["startup"]
                )
            )
        }
        if includeOptionalInputBlocker {
            blockers += [
                closedSurface(
                    "optional.speaker",
                    surface: "optional speaker voice-id",
                    reason: "no bridge-free runtime and assembly evidence covers optional speaker selection yet",
                    gateIDs: ["startup"]
                ),
            ]
        }
        if includeReleaseEvidenceBlockers {
            blockers += [
                closedSurface(
                    "gate.startup-audio",
                    surface: "startup audio gate",
                    reason: "startup audio evidence still crosses release bucket adapter",
                    gateIDs: ["startup"]
                ),
                closedSurface(
                    "gate.audio-contract",
                    surface: "audio contract gate",
                    reason: "audio contract evidence still crosses release bucket adapter",
                    gateIDs: ["audio_contract"]
                ),
                closedSurface(
                    "correctness.stream-parity",
                    surface: "stream parity correctness",
                    reason: "stream parity evidence still crosses target verifier bucket",
                    gateIDs: ["audio_contract"]
                ),
                closedSurface(
                    "release.verify",
                    surface: "release verify",
                    reason: "release verification still crosses target verifier bucket",
                    gateIDs: ["startup", "audio_contract"]
                ),
            ]
        }
        return blockers
    }

    private static func textRuntimeContractEvidence(
        exportGateIDs: [String],
        includeBake: Bool,
        includeLinger: Bool = false,
        run: RuntimeContractHashes,
        bench: RuntimeContractHashes? = nil,
        serve: RuntimeContractHashes? = nil,
        trace: RuntimeContractHashes? = nil,
        bake: RuntimeContractHashes? = nil
    ) -> [RuntimeContractExpectation] {
        var evidence = [
            runtimeContractEvidence(
                surfaceID: "run.text",
                id: "contract-text-run",
                request: .runText,
                gateIDs: exportGateIDs,
                hashes: run
            ),
            runtimeContractEvidence(
                surfaceID: "bench.text",
                id: "contract-text-bench",
                request: .benchDecode,
                gateIDs: ["startup"],
                hashes: bench ?? run
            ),
            runtimeContractEvidence(
                surfaceID: "serve.text",
                id: "contract-text-serve",
                request: .serveText,
                gateIDs: exportGateIDs,
                hashes: serve ?? run
            ),
            runtimeContractEvidence(
                surfaceID: "trace.text",
                id: "contract-text-trace",
                request: .traceTextGenerate,
                gateIDs: exportGateIDs,
                hashes: trace ?? run
            ),
        ]
        if includeBake {
            evidence.append(
                runtimeContractEvidence(
                    surfaceID: "bake.prompt-prefix",
                    id: "contract-text-bake",
                    request: .bakeTextPromptPrefix,
                    gateIDs: exportGateIDs,
                    hashes: bake ?? run
                )
            )
        }
        if includeLinger {
            evidence.append(
                runtimeContractEvidence(
                    surfaceID: "linger.text",
                    id: "contract-text-linger",
                    request: .runText,
                    gateIDs: exportGateIDs,
                    hashes: run
                )
            )
        }
        return evidence
    }

    private static func textRuntimeAssemblyEvidence(
        fixture: String,
        includeBake: Bool = false,
        includeLinger: Bool = false,
        exportGateIDs: [String] = ["startup"]
    ) -> [RuntimeAssemblyExpectation] {
        func evidence(
            surfaceID: String,
            evidenceID: String,
            request: SmeltCAMCapabilityRequest,
            gateIDs: [String]
        ) -> RuntimeAssemblyExpectation {
            let featureContract = try! runtimeAssemblyFeatureContract(
                fixture: fixture,
                request: request
            )
            return RuntimeAssemblyExpectation(
                surfaceID: surfaceID,
                evidenceID: evidenceID,
                request: request,
                exportID: "generate",
                flowID: "generate",
                assemblyPlanKey: "flow-plan-construction",
                executionBackend: "compiled-metal",
                artifactRoles: ["baked-inline", "weights"],
                gateIDs: gateIDs,
                requiredFeatureSet: featureContract.featureSet,
                consumedFeatureSet: featureContract.featureSet,
                unsupportedFeatureSet: [],
                requiredObligationIDs: [],
                consumedObligationIDs: [],
                unsupportedObligationIDs: [],
                selectorDeletionLintID: "cam-block-graph-clean",
                manifestPolicyBridgeAbsent: true
            )
        }
        var rows = [
            evidence(surfaceID: "run.text", evidenceID: "construct-run", request: .runText, gateIDs: exportGateIDs),
            evidence(surfaceID: "bench.text", evidenceID: "construct-bench", request: .benchDecode, gateIDs: ["startup"]),
            evidence(surfaceID: "serve.text", evidenceID: "construct-serve", request: .serveText, gateIDs: exportGateIDs),
            evidence(
                surfaceID: "trace.text",
                evidenceID: "construct-trace",
                request: .traceTextGenerate,
                gateIDs: exportGateIDs
            ),
        ]
        if includeBake {
            rows.insert(
                evidence(
                    surfaceID: "bake.prompt-prefix",
                    evidenceID: "construct-bake",
                    request: .bakeTextPromptPrefix,
                    gateIDs: exportGateIDs
                ),
                at: 3
            )
        }
        if includeLinger {
            rows.append(
                evidence(
                    surfaceID: "linger.text",
                    evidenceID: "construct-linger",
                    request: .runText,
                    gateIDs: exportGateIDs
                )
            )
        }
        return rows
    }

    private static func twoTextRuntimeAssemblyEvidence(
        fixture: String,
        includeLinger: Bool = false
    ) -> [RuntimeAssemblyExpectation] {
        func evidence(
            surfaceID: String,
            evidenceID: String,
            request: SmeltCAMCapabilityRequest
        ) -> RuntimeAssemblyExpectation {
            let featureContract = try! runtimeAssemblyFeatureContract(
                fixture: fixture,
                request: request
            )
            return RuntimeAssemblyExpectation(
                surfaceID: surfaceID,
                evidenceID: evidenceID,
                request: request,
                exportID: "review",
                flowID: "review",
                assemblyPlanKey: "flow-plan-construction",
                executionBackend: "compiled-metal",
                artifactRoles: ["baked-inline", "weights"],
                gateIDs: ["startup"],
                requiredFeatureSet: featureContract.featureSet,
                consumedFeatureSet: featureContract.featureSet,
                unsupportedFeatureSet: [],
                requiredObligationIDs: [],
                consumedObligationIDs: [],
                unsupportedObligationIDs: [],
                selectorDeletionLintID: "cam-block-graph-clean",
                manifestPolicyBridgeAbsent: true
            )
        }
        var rows = [
            evidence(surfaceID: "run.text", evidenceID: "construct-two-input-run", request: twoTextRunRequest),
            evidence(surfaceID: "bench.text", evidenceID: "construct-two-input-bench", request: twoTextBenchRequest),
            evidence(surfaceID: "serve.text", evidenceID: "construct-two-input-serve", request: twoTextServeRequest),
            evidence(
                surfaceID: "bake.prompt-prefix",
                evidenceID: "construct-two-input-bake",
                request: twoTextBakeRequest
            ),
            evidence(surfaceID: "trace.text", evidenceID: "construct-two-input-trace", request: twoTextTraceRequest),
        ]
        if includeLinger {
            rows.append(
                evidence(
                    surfaceID: "linger.text",
                    evidenceID: "construct-two-input-linger",
                    request: twoTextRunRequest
                )
            )
        }
        return rows
    }

    private static func twoTextRuntimeContractEvidence(
        includeLinger: Bool = false,
        run: RuntimeContractHashes,
        bench: RuntimeContractHashes? = nil,
        serve: RuntimeContractHashes? = nil,
        trace: RuntimeContractHashes? = nil,
        bake: RuntimeContractHashes? = nil
    ) -> [RuntimeContractExpectation] {
        var evidence = [
            runtimeContractEvidence(
                surfaceID: "run.text",
                id: "contract-two-input-run",
                request: twoTextRunRequest,
                exportID: "review",
                flowID: "review",
                gateIDs: ["startup"],
                hashes: run
            ),
            runtimeContractEvidence(
                surfaceID: "bench.text",
                id: "contract-two-input-bench",
                request: twoTextBenchRequest,
                exportID: "review",
                flowID: "review",
                gateIDs: ["startup"],
                hashes: bench ?? run
            ),
            runtimeContractEvidence(
                surfaceID: "serve.text",
                id: "contract-two-input-serve",
                request: twoTextServeRequest,
                exportID: "review",
                flowID: "review",
                gateIDs: ["startup"],
                hashes: serve ?? run
            ),
            runtimeContractEvidence(
                surfaceID: "trace.text",
                id: "contract-two-input-trace",
                request: twoTextTraceRequest,
                exportID: "review",
                flowID: "review",
                gateIDs: ["startup"],
                hashes: trace ?? run
            ),
            runtimeContractEvidence(
                surfaceID: "bake.prompt-prefix",
                id: "contract-two-input-bake",
                request: twoTextBakeRequest,
                exportID: "review",
                flowID: "review",
                gateIDs: ["startup"],
                hashes: bake ?? run
            ),
        ]
        if includeLinger {
            evidence.append(
                runtimeContractEvidence(
                    surfaceID: "linger.text",
                    id: "contract-two-input-linger",
                    request: twoTextRunRequest,
                    exportID: "review",
                    flowID: "review",
                    gateIDs: ["startup"],
                    hashes: run
                )
            )
        }
        return evidence
    }

    private static func audioRuntimeContractEvidence(
        exportID: String,
        flowID: String,
        runSurfaceID: String = "run.audio-24khz",
        includeServe: Bool,
        includeBake: Bool,
        includeTrace: Bool,
        includeLinger: Bool = false,
        run: RuntimeContractHashes,
        serve: RuntimeContractHashes? = nil,
        bake: RuntimeContractHashes? = nil,
        trace: RuntimeContractHashes? = nil
    ) -> [RuntimeContractExpectation] {
        var evidence = [
            runtimeContractEvidence(
                surfaceID: runSurfaceID,
                id: "contract-audio-run",
                request: .runAudio,
                exportID: exportID,
                flowID: flowID,
                gateIDs: ["startup"],
                hashes: run
            ),
        ]
        if includeServe {
            evidence.append(
                runtimeContractEvidence(
                    surfaceID: "serve.audio-24khz",
                    id: "contract-audio-serve",
                    request: .serveAudio,
                    exportID: exportID,
                    flowID: flowID,
                    gateIDs: ["startup"],
                    hashes: serve ?? run
                )
            )
        }
        if includeBake {
            evidence.append(
                runtimeContractEvidence(
                    surfaceID: "bake.voice-defaults",
                    id: "contract-audio-bake",
                    request: .bakeVoiceDefaults,
                    exportID: exportID,
                    flowID: flowID,
                    gateIDs: ["startup"],
                    hashes: bake ?? run
                )
            )
        }
        if includeTrace {
            evidence.append(
                runtimeContractEvidence(
                    surfaceID: "trace.audio-24khz",
                    id: "contract-audio-trace",
                    request: .traceTextSynthesize,
                    exportID: exportID,
                    flowID: flowID,
                    gateIDs: ["startup"],
                    hashes: trace ?? run
                )
            )
        }
        if includeLinger {
            evidence.append(
                runtimeContractEvidence(
                    surfaceID: "linger.audio-24khz",
                    id: "contract-audio-linger",
                    request: .runAudio,
                    exportID: exportID,
                    flowID: flowID,
                    gateIDs: ["startup"],
                    hashes: run
                )
            )
        }
        return evidence
    }

    private static func audio24KRuntimeAssemblyEvidence(
        fixture: String,
        exportID: String,
        flowID: String,
        includeServe: Bool,
        includeTrace: Bool = false,
        includeLinger: Bool = false
    ) -> [RuntimeAssemblyExpectation] {
        func evidence(
            surfaceID: String,
            evidenceID: String,
            request: SmeltCAMCapabilityRequest,
            gateIDs: [String] = ["startup"]
        ) -> RuntimeAssemblyExpectation {
            let featureContract = try! runtimeAssemblyFeatureContract(
                fixture: fixture,
                request: request
            )
            return RuntimeAssemblyExpectation(
                surfaceID: surfaceID,
                evidenceID: evidenceID,
                request: request,
                exportID: exportID,
                flowID: flowID,
                assemblyPlanKey: "flow-plan-construction",
                executionBackend: "compiled-metal",
                artifactRoles: ["baked-inline", "sidecar"],
                gateIDs: gateIDs,
                requiredFeatureSet: featureContract.featureSet,
                consumedFeatureSet: featureContract.featureSet,
                unsupportedFeatureSet: [],
                requiredObligationIDs: [],
                consumedObligationIDs: [],
                unsupportedObligationIDs: [],
                selectorDeletionLintID: "cam-block-graph-clean",
                manifestPolicyBridgeAbsent: true
            )
        }
        var rows = [
            evidence(
                surfaceID: "run.audio-24khz",
                evidenceID: "construct-run",
                request: .runAudio
            ),
        ]
        if includeServe {
            rows.append(
                evidence(
                    surfaceID: "serve.audio-24khz",
                    evidenceID: "construct-serve",
                    request: .serveAudio
                )
            )
        }
        if includeLinger {
            rows.append(
                evidence(
                    surfaceID: "linger.audio-24khz",
                    evidenceID: "construct-linger",
                    request: .runAudio
                )
            )
        }
        if includeTrace {
            rows.append(
                evidence(
                    surfaceID: "trace.audio-24khz",
                    evidenceID: "construct-trace",
                    request: .traceTextSynthesize
                )
            )
        }
        return rows
    }

    private static func runtimeContractEvidence(
        surfaceID: String,
        id: String,
        request: SmeltCAMCapabilityRequest,
        exportID: String = "generate",
        flowID: String = "generate",
        gateIDs: [String],
        hashes: RuntimeContractHashes
    ) -> RuntimeContractExpectation {
        RuntimeContractExpectation(
            surfaceID: surfaceID,
            evidenceID: id,
            request: request,
            exportID: exportID,
            flowID: flowID,
            gateIDs: gateIDs,
            bodySHA256: hashes.bodySHA256,
            provenanceSHA256: hashes.provenanceSHA256
        )
    }

    private static func textGenerationExpectations(
        elapsedMs: String,
        bakePromptPrefix: Bool,
        exportGateIDs: [String] = ["startup"],
        includeLinger: Bool = false
    ) -> [CapabilityExpectation] {
        var expectations = [
            CapabilityExpectation(
                request: .runText,
                exportID: "generate",
                flowID: "generate",
                selectedInputs: ["text[encoding=utf8]"],
                selectedOutputs: ["text[encoding=utf8]"],
                gateIDs: exportGateIDs,
                gateRequirements: [],
                gatePredicates: []
            ),
            CapabilityExpectation(
                request: .benchDecode,
                exportID: "generate",
                flowID: "generate",
                selectedInputs: ["text[encoding=utf8]"],
                selectedOutputs: ["text[encoding=utf8]"],
                gateIDs: ["startup"],
                gateRequirements: ["elapsed:<=:\(elapsedMs):ms"],
                gatePredicates: ["tokens:>=:1:none"]
            ),
            CapabilityExpectation(
                request: .serveText,
                exportID: "generate",
                flowID: "generate",
                selectedInputs: ["text[encoding=utf8]"],
                selectedOutputs: ["text[encoding=utf8]"],
                gateIDs: exportGateIDs,
                gateRequirements: [],
                gatePredicates: []
            ),
            CapabilityExpectation(
                request: .traceTextGenerate,
                exportID: "generate",
                flowID: "generate",
                selectedInputs: ["text[encoding=utf8]"],
                selectedOutputs: ["text[encoding=utf8]"],
                gateIDs: exportGateIDs,
                gateRequirements: [],
                gatePredicates: []
            ),
        ]
        if bakePromptPrefix {
            expectations.append(
                CapabilityExpectation(
                    request: .bakeTextPromptPrefix,
                    exportID: "generate",
                    flowID: "generate",
                    selectedInputs: ["text[encoding=utf8]"],
                    selectedOutputs: ["text[encoding=utf8]"],
                    gateIDs: exportGateIDs,
                    gateRequirements: [],
                    gatePredicates: []
                )
            )
        }
        if includeLinger {
            expectations.append(
                CapabilityExpectation(
                    request: .runText,
                    exportID: "generate",
                    flowID: "generate",
                    selectedInputs: ["text[encoding=utf8]"],
                    selectedOutputs: ["text[encoding=utf8]"],
                    gateIDs: exportGateIDs,
                    gateRequirements: [],
                    gatePredicates: [],
                    surfaceID: "linger.text"
                )
            )
        }
        return expectations
    }

    private static func twoTextGenerationExpectations(elapsedMs: String) -> [CapabilityExpectation] {
        func expectation(
            _ request: SmeltCAMCapabilityRequest,
            surfaceID: String,
            gateRequirements: [String] = [],
            gatePredicates: [String] = []
        ) -> CapabilityExpectation {
            CapabilityExpectation(
                request: request,
                exportID: "review",
                flowID: "review",
                selectedInputs: ["text[encoding=utf8]", "text[encoding=utf8]"],
                selectedOutputs: ["text[encoding=utf8]"],
                selectedInputNames: ["candidate", "context"],
                selectedOutputNames: ["text"],
                gateIDs: ["startup"],
                gateRequirements: gateRequirements,
                gatePredicates: gatePredicates,
                surfaceID: surfaceID
            )
        }

        return [
            expectation(twoTextRunRequest, surfaceID: "run.text"),
            expectation(
                twoTextBenchRequest,
                surfaceID: "bench.text",
                gateRequirements: ["elapsed:<=:\(elapsedMs):ms"],
                gatePredicates: ["tokens:>=:1:none"]
            ),
            expectation(twoTextServeRequest, surfaceID: "serve.text"),
            expectation(twoTextTraceRequest, surfaceID: "trace.text"),
            expectation(twoTextBakeRequest, surfaceID: "bake.prompt-prefix"),
            expectation(twoTextRunRequest, surfaceID: "linger.text"),
        ]
    }

    private static func textToAudio24KExpectations() -> [CapabilityExpectation] {
        textToAudioRunExpectations(exportID: "synth", flowID: "synth", rate: "24khz")
        + [
            .init(
                request: .serveAudio,
                exportID: "synth",
                flowID: "synth",
                selectedInputs: ["text[encoding=utf8]"],
                selectedOutputs: ["pcm[dtype=f32,rate=24khz]"],
                gateIDs: ["startup"],
                gateRequirements: ["elapsed:<=:400:ms"],
                gatePredicates: [
                    "duration:>=:20:ms",
                    "format:==:pcm f32 24khz:none",
                ]
            ),
            .init(
                request: .traceTextSynthesize,
                exportID: "synth",
                flowID: "synth",
                selectedInputs: ["text[encoding=utf8]"],
                selectedOutputs: ["pcm[dtype=f32,rate=24khz]"],
                gateIDs: ["startup"],
                gateRequirements: ["elapsed:<=:400:ms"],
                gatePredicates: [
                    "duration:>=:20:ms",
                    "format:==:pcm f32 24khz:none",
                ]
            ),
            .init(
                request: .runAudio,
                exportID: "synth",
                flowID: "synth",
                selectedInputs: ["text[encoding=utf8]"],
                selectedOutputs: ["pcm[dtype=f32,rate=24khz]"],
                selectedInputNames: ["text"],
                gateIDs: ["startup"],
                gateRequirements: ["elapsed:<=:400:ms"],
                gatePredicates: [
                    "duration:>=:20:ms",
                    "format:==:pcm f32 24khz:none",
                ],
                surfaceID: "linger.audio-24khz"
            ),
        ]
    }

    private static func textToAudioRunExpectations(
        exportID: String,
        flowID: String,
        rate: String
    ) -> [CapabilityExpectation] {
        [
            .init(
                request: .runAudio,
                exportID: exportID,
                flowID: flowID,
                selectedInputs: ["text[encoding=utf8]"],
                selectedOutputs: ["pcm[dtype=f32,rate=\(rate)]"],
                gateIDs: ["startup"],
                gateRequirements: ["elapsed:<=:400:ms"],
                gatePredicates: [
                    "duration:>=:20:ms",
                    "format:==:pcm f32 \(rate):none",
                ]
            ),
        ]
    }
}
