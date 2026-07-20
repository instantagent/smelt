import Foundation
import XCTest
@testable import SmeltRuntime
import SmeltSchema

final class Qwen3TTSAudioTests: XCTestCase {

    func testPcm16ClampsAndScales() {
        let pcm = Qwen3TTSAudio.pcm16([0, 1, -1, 2, -2, 0.5])
        let vals = pcm.withUnsafeBytes { Array($0.bindMemory(to: Int16.self)) }
            .map { Int16(littleEndian: $0) }
        XCTAssertEqual(vals[0], 0)
        XCTAssertEqual(vals[1], 32767)
        XCTAssertEqual(vals[2], -32767, "symmetric scale: -1 maps to -32767, not -32768")
        XCTAssertEqual(vals[3], 32767, "out-of-range clamps")
        XCTAssertEqual(vals[4], -32767)
        XCTAssertEqual(vals[5], 16384, "0.5 * 32767 rounds to 16384")
    }

    func testPcm16NonFiniteEncodesAsSilence() {
        let pcm = Qwen3TTSAudio.pcm16([.nan, .infinity, -.infinity])
        let vals = pcm.withUnsafeBytes { Array($0.bindMemory(to: Int16.self)) }
            .map { Int16(littleEndian: $0) }
        XCTAssertEqual(vals, [0, 0, 0], "NaN/inf must not become full-scale clicks")
    }

    func testWavHeaderAndPayload() {
        let samples: [Float] = Array(repeating: 0.25, count: 480)
        let wav = Qwen3TTSAudio.wav(samples)
        XCTAssertEqual(wav.count, 44 + 960, "44-byte header + 2 bytes/sample")
        XCTAssertEqual(String(data: wav[0..<4], encoding: .ascii), "RIFF")
        XCTAssertEqual(String(data: wav[8..<12], encoding: .ascii), "WAVE")
        XCTAssertEqual(String(data: wav[12..<16], encoding: .ascii), "fmt ")
        XCTAssertEqual(String(data: wav[36..<40], encoding: .ascii), "data")
        func u32(_ off: Int) -> UInt32 {
            wav.subdata(in: off..<(off + 4)).withUnsafeBytes { UInt32(littleEndian: $0.load(as: UInt32.self)) }
        }
        func u16(_ off: Int) -> UInt16 {
            wav.subdata(in: off..<(off + 2)).withUnsafeBytes { UInt16(littleEndian: $0.load(as: UInt16.self)) }
        }
        XCTAssertEqual(u32(4), UInt32(36 + 960), "RIFF size")
        XCTAssertEqual(u16(20), 1, "PCM format tag")
        XCTAssertEqual(u16(22), 1, "mono")
        XCTAssertEqual(u32(24), 24000, "sample rate")
        XCTAssertEqual(u32(28), 48000, "byte rate = rate * 2")
        XCTAssertEqual(u16(32), 2, "block align")
        XCTAssertEqual(u16(34), 16, "bits per sample")
        XCTAssertEqual(u32(40), 960, "data chunk size")
        XCTAssertEqual(wav.suffix(from: 44), Qwen3TTSAudio.pcm16(samples), "payload is the pcm16 bytes")
    }

    func testStreamingWavHeaderAndPatches() {
        let header = Qwen3TTSAudio.wavHeader(pcmBytes: nil)
        XCTAssertEqual(header.count, 44)
        func u32(_ d: Data, _ off: Int) -> UInt32 {
            d.subdata(in: off..<(off + 4)).withUnsafeBytes { UInt32(littleEndian: $0.load(as: UInt32.self)) }
        }
        XCTAssertEqual(u32(header, 4), .max, "unknown-length RIFF size sentinel")
        XCTAssertEqual(u32(header, 40), .max, "unknown-length data size sentinel")

        // Applying the patches to a streamed header must reproduce the exact header.
        var patched = header
        for p in Qwen3TTSAudio.wavSizePatches(pcmBytes: 960) {
            var v = p.value.littleEndian
            withUnsafeBytes(of: &v) { patched.replaceSubrange(p.offset..<(p.offset + 4), with: $0) }
        }
        XCTAssertEqual(patched, Qwen3TTSAudio.wavHeader(pcmBytes: 960))
    }

    func testVoiceRoundTripAndAbsence() throws {
        let dir = NSTemporaryDirectory() + "voice-test-\(UUID().uuidString)"
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(atPath: dir) }

        XCTAssertNil(try Qwen3TTSVoice.load(packagePath: dir), "absent voice.json loads as nil")

        let voice = Qwen3TTSVoice(speaker: "vivian", language: "English", instruct: nil,
                                  firstChunkFrames: 1, maxChunkFrames: 1, maxFrames: 256)
        try voice.write(packagePath: dir)
        XCTAssertEqual(try Qwen3TTSVoice.load(packagePath: dir), voice)

        // Malformed file must throw, not silently fall back to defaults.
        try Data("not json".utf8).write(to: URL(fileURLWithPath: dir + "/voice.json"))
        XCTAssertThrowsError(try Qwen3TTSVoice.load(packagePath: dir))
    }

    func testVoiceSemanticValidation() throws {
        let dir = NSTemporaryDirectory() + "voice-test-\(UUID().uuidString)"
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(atPath: dir) }

        func writeAndLoad(_ voice: Qwen3TTSVoice) throws {
            try voice.write(packagePath: dir)
            _ = try Qwen3TTSVoice.load(packagePath: dir)
        }

        XCTAssertThrowsError(try writeAndLoad(Qwen3TTSVoice(
            firstChunkFrames: 0, maxChunkFrames: 1, maxFrames: 256
        ))) {
            XCTAssertTrue(String(describing: $0).contains("firstChunkFrames"))
        }
        XCTAssertThrowsError(try writeAndLoad(Qwen3TTSVoice(
            firstChunkFrames: 2, maxChunkFrames: 1, maxFrames: 256
        ))) {
            XCTAssertTrue(String(describing: $0).contains("maxChunkFrames"))
        }
        XCTAssertThrowsError(try writeAndLoad(Qwen3TTSVoice(maxFrames: 0))) {
            XCTAssertTrue(String(describing: $0).contains("maxFrames"))
        }
        XCTAssertThrowsError(try writeAndLoad(Qwen3TTSVoice(language: "  "))) {
            XCTAssertTrue(String(describing: $0).contains("language"))
        }
    }
}
