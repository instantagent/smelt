import Foundation
import SmeltModels
import SmeltModuleAuthoring
import SmeltSchema
import XCTest

/// Determinism / regression gate for the Swift-authored model definitions.
///
/// The `.cam` text grammar — the former parity oracle — was deleted in Phase C
/// of the module-authoring migration. The checked-in `Models/<id>.module.json`
/// files are now the sole canonical artifact, emitted by `smelt-models emit`
/// from `SmeltModels.all`. This gate pins that the authoring code still emits
/// exactly what is committed: for every definition, its canonical JSON
/// (`prettyPrinted: true`, the emit format) must byte-equal the checked-in
/// `.module.json`, so a stray authoring edit or a forgotten regenerate is a
/// hard failure. Byte equality localizes divergence for debugging and subsumes
/// the `semanticSHA256` / `exportABISHA256` pins, which are re-asserted here
/// against the committed file as an explicit canonicalization cross-check.
final class ModuleAuthoringParityTests: XCTestCase {
    private static var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private func moduleJSONURL(_ id: String) -> URL {
        Self.repoRoot
            .appendingPathComponent("Models", isDirectory: true)
            .appendingPathComponent("\(id).module.json")
    }

    func testAuthoredDefinitionsMatchCheckedInModuleJSON() throws {
        for definition in SmeltModels.all {
            let id = definition.module.id
            let url = moduleJSONURL(id)
            // Reproduce the exact on-disk artifact `smelt-models emit` writes:
            // canonical pretty JSON followed by a single trailing newline.
            var authored = try definition.canonicalJSONData(prettyPrinted: true)
            authored.append(0x0A)
            let committed = try Data(contentsOf: url)

            if authored != committed {
                let authoredLines = String(decoding: authored, as: UTF8.self)
                    .split(separator: "\n", omittingEmptySubsequences: false)
                let committedLines = String(decoding: committed, as: UTF8.self)
                    .split(separator: "\n", omittingEmptySubsequences: false)
                let firstDiff = zip(authoredLines, committedLines).enumerated()
                    .first { $0.element.0 != $0.element.1 }
                let hint = firstDiff.map {
                    "line \($0.offset + 1):\n  authored:  \($0.element.0)\n  committed: \($0.element.1)"
                } ?? "line counts differ (authored \(authoredLines.count), committed \(committedLines.count))"
                XCTFail("""
                \(id): authored IR diverges from checked-in Models/\(id).module.json — \
                re-run `smelt-models emit --output Models`.
                \(hint)
                """)
            }

            // Byte equality already implies hash equality; re-assert the pins
            // against the committed file so a canonicalization/decode regression
            // surfaces distinctly from a raw-bytes diff.
            let committedIR = try SmeltCAMIR.decodeModule(at: url)
            XCTAssertEqual(
                try definition.semanticSHA256(), try committedIR.semanticSHA256(),
                "\(id): semantic hash")
            XCTAssertEqual(
                try definition.exportABISHA256(), try committedIR.exportABISHA256(),
                "\(id): export ABI hash")
        }
    }

    func testXMLToolTranscriptCodecIsIndependentFromChatTemplate() throws {
        for id in [
            "qwen35_fast", "qwen35_text", "qwen35_4b", "qwen36_27b",
            "bonsai_27b_binary", "bonsai_27b_ternary",
        ] {
            let module = try XCTUnwrap(SmeltModels.definition(id: id))
            let tokenizer = try XCTUnwrap(
                module.graphNodes.first { $0.id == "tokenizer" }
            )
            let annotations = Dictionary(
                uniqueKeysWithValues: tokenizer.annotations.map { ($0.key, $0.value) }
            )
            XCTAssertEqual(annotations["prompt-format"], SmeltPromptTemplateName.chatML)
            XCTAssertEqual(
                annotations["tool-format"],
                SmeltToolTranscriptCodecName.xmlFunctionParameters
            )
        }

        let qwen35FourB = try XCTUnwrap(SmeltModels.definition(id: "qwen35_4b"))
        let multimodalTokenizer = try XCTUnwrap(
            qwen35FourB.graphNodes.first { $0.id == "multimodal-tokenizer" }
        )
        let multimodalAnnotations = Dictionary(
            uniqueKeysWithValues: multimodalTokenizer.annotations.map { ($0.key, $0.value) }
        )
        XCTAssertEqual(
            multimodalAnnotations["tool-format"],
            SmeltToolTranscriptCodecName.xmlFunctionParameters
        )
    }
}
