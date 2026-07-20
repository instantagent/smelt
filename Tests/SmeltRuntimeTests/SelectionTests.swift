import XCTest
@testable import SmeltRuntime

final class SelectionTests: XCTestCase {
    func testArgmaxSelectionUsesMaximumLogit() {
        let logits: [Float] = [-2.0, 0.5, 3.25, 1.0]
        XCTAssertEqual(
            SmeltLogitsSelector.select(
                logits: logits,
                position: 0,
                mode: .argmax
            ),
            2
        )
    }

    func testTemperatureZeroFallsBackToArgmax() {
        let logits: [Float] = [0.25, 1.5, -0.75, 1.25]
        let greedy = SmeltLogitsSelector.select(
            logits: logits,
            position: 17,
            mode: .argmax
        )
        let sampled = SmeltLogitsSelector.select(
            logits: logits,
            position: 17,
            mode: .temperature(0, seed: 123)
        )
        XCTAssertEqual(sampled, greedy)
    }

    func testTemperatureSelectionIsDeterministicForSameSeedAndPosition() {
        let logits: [Float] = [0.0, 0.5, -0.25, 1.0, 0.75, -1.0]
        let first = SmeltLogitsSelector.select(
            logits: logits,
            position: 42,
            mode: .temperature(0.8, seed: 999)
        )
        let second = SmeltLogitsSelector.select(
            logits: logits,
            position: 42,
            mode: .temperature(0.8, seed: 999)
        )
        XCTAssertEqual(first, second)
    }

    func testTemperatureSelectionReturnsTokenInRange() {
        let logits = Array(repeating: Float(0), count: 32)
        let token = SmeltLogitsSelector.select(
            logits: logits,
            position: 9,
            mode: .temperature(1.0, seed: 12345)
        )
        XCTAssertGreaterThanOrEqual(token, 0)
        XCTAssertLessThan(Int(token), logits.count)
    }

    func testArgmaxSelectionHonorsAllowedTokenMask() {
        let logits: [Float] = [-1.0, 2.0, 99.0, 4.0]
        let token = SmeltLogitsSelector.select(
            logits: logits,
            position: 0,
            mode: .argmax,
            allowedTokenMask: mask(allowing: [1, 3], vocabSize: logits.count)
        )
        XCTAssertEqual(token, 3)
    }

    func testTemperatureSelectionHonorsAllowedTokenMask() {
        let logits: [Float] = [10.0, 9.0, 8.0, 7.0]
        let allowed = [2, 3]
        let token = SmeltLogitsSelector.select(
            logits: logits,
            position: 5,
            mode: .temperature(1.0, seed: 42),
            allowedTokenMask: mask(allowing: allowed, vocabSize: logits.count)
        )
        XCTAssertTrue(allowed.contains(Int(token)))
    }

    func testFilteredTemperatureNeverLeavesTopK() {
        let logits: [Float] = [10, 9, 8, 7, 6]
        for position in 0..<64 {
            let token = SmeltLogitsSelector.select(
                logits: logits,
                position: Int32(position),
                mode: .filteredTemperature(
                    0.7, topK: 2, topP: 1, seed: 42
                )
            )
            XCTAssertTrue([0, 1].contains(Int(token)))
        }
    }

    func testFilteredTemperatureAppliesGrammarMaskBeforeTopK() {
        let logits: [Float] = [100, 90, 8, 7, 6]
        let allowed = [2, 3, 4]
        for position in 0..<64 {
            let token = SmeltLogitsSelector.select(
                logits: logits,
                position: Int32(position),
                mode: .filteredTemperature(
                    0.7, topK: 2, topP: 0.95, seed: 7
                ),
                allowedTokenMask: mask(
                    allowing: allowed, vocabSize: logits.count
                )
            )
            XCTAssertTrue([2, 3].contains(Int(token)))
        }
    }

    func testFilteredTemperatureTopPIncludesBoundaryToken() {
        // Equal logits have equal mass. top_p=0.6 keeps the smallest
        // descending prefix reaching 60%: three of four candidates.
        let logits: [Float] = [1, 1, 1, 1]
        for position in 0..<128 {
            let token = SmeltLogitsSelector.select(
                logits: logits,
                position: Int32(position),
                mode: .filteredTemperature(
                    1, topK: nil, topP: 0.6, seed: 99
                )
            )
            XCTAssertTrue([0, 1, 2].contains(Int(token)))
        }
    }

    private func mask(allowing tokenIds: [Int], vocabSize: Int) -> [UInt32] {
        var mask = [UInt32](repeating: 0, count: (vocabSize + 31) / 32)
        for tokenId in tokenIds {
            mask[tokenId / 32] |= UInt32(1) << UInt32(tokenId % 32)
        }
        return mask
    }
}
