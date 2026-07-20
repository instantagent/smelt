import Foundation
import Testing
import SmeltRuntime

@Suite struct SmeltOrderedJSONTests {
    @Test func retainsNestedToolSchemaAndStrictFieldOrder() throws {
        let source = #"{"type":"function","function":{"name":"read","description":"Read","parameters":{"type":"object","required":["path"],"properties":{"path":{"type":"string","description":"Path"},"offset":{"type":"number"}}},"strict":false}}"#
        let value = try SmeltOrderedJSONValue.parse(source)
        #expect(value.compactJSON == source)
        #expect(value.templateJSON == #"{"type": "function", "function": {"name": "read", "description": "Read", "parameters": {"type": "object", "required": ["path"], "properties": {"path": {"type": "string", "description": "Path"}, "offset": {"type": "number"}}}, "strict": false}}"#)
    }

    @Test func rejectsTrailingAndMalformedJSON() {
        #expect(throws: SmeltOrderedJSONError.self) {
            try SmeltOrderedJSONValue.parse(#"{"a":1} nope"#)
        }
        #expect(throws: SmeltOrderedJSONError.self) {
            try SmeltOrderedJSONValue.parse(#"{"a":01}"#)
        }
    }
}
