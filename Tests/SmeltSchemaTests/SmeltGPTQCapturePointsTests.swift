import Foundation
import Testing

@testable import SmeltSchema

@Suite("Smelt GPTQ capture points")
struct SmeltGPTQCapturePointsTests {
    @Test("Canonical package metadata has stable sorted keys")
    func canonicalJSON() throws {
        let value = SmeltGPTQCapturePoints(prefill: [
            SmeltGPTQCapturePoint(
                weightName: "layer.weight",
                inputSlot: 8,
                k: 896,
                dispatchCount: 2,
                inputIsFloat16: false
            ),
        ])
        let text = try #require(
            String(data: value.canonicalJSONData(), encoding: .utf8)
        )
        #expect(
            text
                == #"{"prefill":[{"dispatchCount":2,"inputIsFloat16":false,"inputSlot":8,"k":896,"weightName":"layer.weight"}]}"#
        )
    }
}
