// SmeltAudioOutput — neutral audio-output helpers shared by the audio run
// surfaces. Parameterized by sample rate + channel count so one encoder serves
// 24 kHz mono (TTS) or any other rate/channel layout. Input is
// CHANNEL-MAJOR ([ch0 frame0..frameN][ch1 frame0..frameN]); WAV output is
// frame-interleaved 16-bit little-endian PCM in a RIFF/WAVE container.

import Foundation

public enum SmeltAudioOutput {
    /// A complete WAV file (44-byte RIFF header + interleaved Int16 PCM) for the
    /// channel-major `samples`. `channels == 1` makes the interleave an identity.
    /// Non-finite samples are written as silence; values are clamped to [-1, 1].
    public static func wavData(
        channelMajor samples: [Float], sampleRate: Int, channels: Int
    ) -> Data {
        let pcm = pcm16Data(channelMajor: samples, sampleRate: sampleRate, channels: channels)
        return riff(pcm: pcm, sampleRate: sampleRate, channels: channels)
    }

    /// Interleaved little-endian Int16 PCM for channel-major Float samples.
    public static func pcm16Data(
        channelMajor samples: [Float], sampleRate: Int, channels: Int
    ) -> Data {
        precondition(
            sampleRate > 0 && channels > 0 && samples.count % channels == 0,
            "pcm16Data: \(samples.count) samples, \(channels) channels, \(sampleRate) Hz")
        let frames = samples.count / channels
        var pcm = Data(count: samples.count * MemoryLayout<Int16>.stride)
        pcm.withUnsafeMutableBytes { raw in
            let dst = raw.bindMemory(to: Int16.self)
            for frame in 0..<frames {
                for channel in 0..<channels {
                    let s = samples[channel * frames + frame]
                    let v = s.isFinite ? max(-1, min(1, s)) * 32767 : 0
                    dst[frame * channels + channel] = Int16(v.rounded()).littleEndian
                }
            }
        }
        return pcm
    }

    private static func riff(pcm: Data, sampleRate: Int, channels: Int) -> Data {
        var data = Data(capacity: 44 + pcm.count)
        func u32(_ value: UInt32) {
            var v = value.littleEndian
            withUnsafeBytes(of: &v) { data.append(contentsOf: $0) }
        }
        func u16(_ value: UInt16) {
            var v = value.littleEndian
            withUnsafeBytes(of: &v) { data.append(contentsOf: $0) }
        }
        let ch = UInt16(channels)
        let bits: UInt16 = 16
        data.append(contentsOf: Array("RIFF".utf8))
        u32(UInt32(36 + pcm.count))
        data.append(contentsOf: Array("WAVE".utf8))
        data.append(contentsOf: Array("fmt ".utf8))
        u32(16)
        u16(1)  // PCM
        u16(ch)
        u32(UInt32(sampleRate))
        u32(UInt32(sampleRate) * UInt32(ch) * UInt32(bits / 8))  // byte rate
        u16(ch * bits / 8)  // block align
        u16(bits)
        data.append(contentsOf: Array("data".utf8))
        u32(UInt32(pcm.count))
        data.append(pcm)
        return data
    }
}
