// Gates for allowedTokenMask on the plain SmeltModel.generate paths (the
// surface `smelt run` uses for baked grammars): a JSON-schema matcher must
// constrain output to parseable, schema-shaped JSON, including through the
// baked-prefix restore path. Gated on a locally built package.

import Foundation
import Testing
@testable import SmeltRuntime

private let qwen08bPackage =
    "artifacts/qwen35-0.8b-qmm16x128/Qwen_Qwen3.5-0.8B.smeltpkg"

private let schema = """
    {
      "type": "object",
      "properties": {
        "answer": {"type": "string", "maxLength": 80}
      },
      "required": ["answer"],
      "additionalProperties": false
    }
    """

@Suite struct ConstrainedGenerateTests {
    @Test func maskedGenerateProducesSchemaConformantJSON() throws {
        guard FileManager.default.fileExists(atPath: qwen08bPackage) else { return }
        let package = try hardlinkClone(of: qwen08bPackage)
        let tokenizer = try SmeltTokenizer(path: "\(package)/tokenizer.json")
        let model = try SmeltModel(package: package)
        let llgTokenizer = try SmeltLLGuidanceTokenizer(tokenizer: tokenizer)
        let prototype = try SmeltLLGuidanceMatcher(
            tokenizer: llgTokenizer, jsonSchema: schema
        )

        let ids = tokenizer.encodeWithSpecials(
            "<|im_start|>user\nName one primary color.<|im_end|>\n"
                + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )

        let matcher = try prototype.freshCopy()
        var count = 0
        let result = try model.generate(
            tokenIds: ids,
            allowedTokenMask: { try matcher.computeMask() }
        ) { token in
            count += 1
            guard (try? matcher.consume(tokenIds: [token.id])) != nil else {
                return false
            }
            if matcher.isAccepting { return false }
            return count < 96
        }

        let text = tokenizer.decode(result.tokens)
        let object = try JSONSerialization.jsonObject(
            with: Data(text.utf8)
        ) as? [String: Any]
        #expect(object?.keys.sorted() == ["answer"], "got: \(text)")
        #expect(object?["answer"] as? String != nil)
    }
}
