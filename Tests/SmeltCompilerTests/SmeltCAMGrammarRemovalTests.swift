import Foundation
import Testing
@testable import SmeltCompiler

/// Fail-closed coverage for the deleted `.cam` authoring grammar (Phase C).
///
/// Two layers: (1) the `SmeltCAMGrammarRemoval.rejectionDiagnostic` gateway
/// itself rejects a `.cam` path with the deleted-format message and passes
/// everything else through; (2) a source lint proving every command that used
/// to accept a `.cam` input — `smelt build` and `smelt module check|admission|ir` —
/// routes its input path through that gateway, so
/// none can regress to a confusing JSON-decode error or a silent fallback.
@Suite struct SmeltCAMGrammarRemovalTests {
    @Test func rejectsCamGrammarInputWithDeletedFormatDiagnostic() throws {
        let diagnostic = try #require(
            SmeltCAMGrammarRemoval.rejectionDiagnostic(forInputPath: "Examples/CAM/qwen35_text.cam")
        )
        #expect(diagnostic.contains("the .cam authoring grammar was removed"))
        #expect(diagnostic.contains(".module.json"))
        #expect(diagnostic.contains("qwen35_text.cam"))
    }

    @Test func acceptsModuleJSONAndOtherInputs() {
        #expect(SmeltCAMGrammarRemoval.rejectionDiagnostic(
            forInputPath: "Models/qwen35_text.module.json") == nil)
        #expect(SmeltCAMGrammarRemoval.rejectionDiagnostic(
            forInputPath: "model.module.json") == nil)
        #expect(SmeltCAMGrammarRemoval.rejectionDiagnostic(
            forInputPath: "weights.bin") == nil)
        // A path that merely CONTAINS "cam" but does not end in .cam passes.
        #expect(SmeltCAMGrammarRemoval.rejectionDiagnostic(
            forInputPath: "camera/model.module.json") == nil)
    }

    @Test func everyCamAcceptingCommandRoutesThroughTheGateway() throws {
        // Each command whose input path could be a legacy `.cam` file must
        // consult the gateway before decoding. A source scan is the honest
        // check here: SmeltCLI has no in-process test target, and the gateway's
        // behavior is unit-tested above, so proving each command references it
        // establishes fail-closed wiring end to end.
        let commandFiles = [
            "Sources/SmeltCLI/Commands/Build.swift",
            "Sources/SmeltCLI/Commands/CAM.swift",
        ]
        for relPath in commandFiles {
            let source = try String(contentsOfFile: try repoRelativePath(relPath), encoding: .utf8)
            #expect(
                source.contains("SmeltCAMGrammarRemoval.rejectionDiagnostic"),
                Comment(rawValue: "\(relPath) does not route its input through the .cam fail-closed gateway")
            )
        }
    }
}
