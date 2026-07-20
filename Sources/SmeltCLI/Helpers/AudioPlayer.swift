// SmeltAudioPlayer — live speaker playback via AVAudioEngine, parameterized by
// sample rate + channel count, used by the TTS run surface.
// Input frames are CHANNEL-MAJOR ([ch0 frame0..frameN][ch1 frame0..frameN]),
// which maps directly onto AVAudioPCMBuffer's planar `floatChannelData[ch]` (no
// interleave needed for playback — interleaving is only for WAV/PCM output).

import AVFoundation

final class SmeltAudioPlayer {
    private let engine = AVAudioEngine()
    private let node = AVAudioPlayerNode()
    private let format: AVAudioFormat
    private let channels: Int
    private let group = DispatchGroup()

    init(sampleRate: Int, channels: Int) throws {
        guard channels > 0, let fmt = AVAudioFormat(
            standardFormatWithSampleRate: Double(sampleRate),
            channels: AVAudioChannelCount(channels)
        ) else {
            throw CLIError("audio output init failed (\(sampleRate) Hz, \(channels) ch)")
        }
        self.format = fmt
        self.channels = channels
        engine.attach(node)
        engine.connect(node, to: engine.mainMixerNode, format: format)
        try engine.start()
        node.play()
    }

    /// Enqueue channel-major `samples` for playback.
    func enqueue(channelMajor samples: [Float]) {
        guard channels > 0, samples.count % channels == 0 else { return }
        let frames = samples.count / channels
        guard frames > 0,
              let buf = AVAudioPCMBuffer(
                pcmFormat: format, frameCapacity: AVAudioFrameCount(frames))
        else { return }
        buf.frameLength = AVAudioFrameCount(frames)
        samples.withUnsafeBufferPointer { src in
            for channel in 0..<channels {
                buf.floatChannelData![channel].update(
                    from: src.baseAddress! + channel * frames, count: frames)
            }
        }
        group.enter()
        node.scheduleBuffer(buf) { [group] in group.leave() }
    }

    /// Block until every scheduled buffer has played out, then stop the engine.
    func finish() {
        group.wait()
        node.stop()
        engine.stop()
    }
}
