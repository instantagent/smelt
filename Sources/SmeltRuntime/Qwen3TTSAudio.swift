// Qwen3TTSAudio — wire/file encodings for the TTS pipeline's 24 kHz mono Float32
// samples: 16-bit little-endian PCM (the streaming wire format — every consumer
// speaks it and it halves the bytes of f32) and a RIFF/WAVE container around it
// (the file/offline format). Shared by text-to-PCM run and serve surfaces.

import Foundation

public enum Qwen3TTSAudio {
    public static let sampleRate = 24_000

    /// Float32 [-1, 1] → 16-bit little-endian PCM bytes (clamped, symmetric 32767 scale).
    /// Non-finite samples encode as silence — Swift's min/max would otherwise pass NaN
    /// through to full-scale (a loud click masking the upstream numeric fault).
    public static func pcm16(_ samples: [Float]) -> Data {
        var out = Data(count: samples.count * 2)
        out.withUnsafeMutableBytes { raw in
            let dst = raw.bindMemory(to: Int16.self)
            for (i, s) in samples.enumerated() {
                let v = s.isFinite ? max(-1, min(1, s)) * 32767 : 0
                dst[i] = Int16(v.rounded()).littleEndian
            }
        }
        return out
    }

    /// 16-bit PCM mono RIFF/WAVE file bytes.
    public static func wav(_ samples: [Float]) -> Data {
        let pcm = pcm16(samples)
        return wavHeader(pcmBytes: pcm.count) + pcm
    }

    /// The 44-byte RIFF/WAVE header for `pcmBytes` of 16-bit mono PCM. `nil` = unknown
    /// length (streaming to a pipe): both size fields carry 0xFFFFFFFF, the streaming-WAV
    /// convention players/converters treat as "read until EOF"; a seekable writer patches
    /// the real sizes afterwards (`wavSizePatches`).
    public static func wavHeader(pcmBytes: Int?) -> Data {
        var out = Data(capacity: 44)
        func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { out.append(contentsOf: $0) } }
        func u16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { out.append(contentsOf: $0) } }
        let rate = UInt32(sampleRate), channels: UInt16 = 1, bits: UInt16 = 16
        let byteRate = rate * UInt32(channels) * UInt32(bits / 8)
        out.append(contentsOf: Array("RIFF".utf8)); u32(pcmBytes.map { UInt32(36 + $0) } ?? .max)
        out.append(contentsOf: Array("WAVE".utf8))
        out.append(contentsOf: Array("fmt ".utf8)); u32(16)
        u16(1) /* PCM */; u16(channels); u32(rate); u32(byteRate)
        u16(channels * bits / 8) /* block align */; u16(bits)
        out.append(contentsOf: Array("data".utf8)); u32(pcmBytes.map { UInt32($0) } ?? .max)
        return out
    }

    /// (byte offset, little-endian u32 value) pairs that turn a streamed unknown-length
    /// header into an exact one once `pcmBytes` is known — for seekable outputs.
    public static func wavSizePatches(pcmBytes: Int) -> [(offset: Int, value: UInt32)] {
        [(4, UInt32(36 + pcmBytes)), (40, UInt32(pcmBytes))]
    }
}
