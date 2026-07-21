// Qwen3TTSVoice - the prepared voice-defaults sidecar (`voice.json`) for a Qwen3-TTS
// `.smeltpkg`. Package preparation writes it; text-to-PCM run and serve surfaces read
// it as per-package defaults so a package can be self-contained (model + persona +
// streaming schedule) and run with no flags. Every field is optional: an explicit
// CLI flag / request field overrides the prepared value, which
// overrides the built-in default.

import Foundation

public enum Qwen3TTSVoiceError: Error, CustomStringConvertible, Equatable {
    case emptyField(String)
    case invalidSchedule(String)

    public var description: String {
        switch self {
        case .emptyField(let field):
            return "voice.json field '\(field)' is present but empty"
        case .invalidSchedule(let reason):
            return "voice.json has invalid streaming schedule: \(reason)"
        }
    }
}

public struct Qwen3TTSVoice: Codable, Sendable, Equatable {
    /// Named speaker from the package's spk_id table (CustomVoice packages).
    public let speaker: String?
    /// Language key from the package's codec_language_id table (e.g. "English", "Auto").
    public let language: String?
    /// Voice-design / style instruction (VoiceDesign packages; styling on CustomVoice 1.7B).
    public let instruct: String?
    /// Streaming chunk schedule (generateStreaming firstChunkFrames/maxChunkFrames).
    /// 1/1 = minimum-TTFA zero-buffer voice-turn schedule; 4/4 = long-form default.
    public let firstChunkFrames: Int?
    public let maxChunkFrames: Int?
    /// Generation cap in 80ms codec frames (response-length budget).
    public let maxFrames: Int?

    public init(speaker: String? = nil, language: String? = nil, instruct: String? = nil,
                firstChunkFrames: Int? = nil, maxChunkFrames: Int? = nil, maxFrames: Int? = nil) {
        self.speaker = speaker
        self.language = language
        self.instruct = instruct
        self.firstChunkFrames = firstChunkFrames
        self.maxChunkFrames = maxChunkFrames
        self.maxFrames = maxFrames
    }

    public static let fileName = "voice.json"

    public func validate() throws {
        func requireNonEmpty(_ value: String?, _ field: String) throws {
            guard let value else { return }
            if value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                throw Qwen3TTSVoiceError.emptyField(field)
            }
        }

        try requireNonEmpty(speaker, "speaker")
        try requireNonEmpty(language, "language")
        if let firstChunkFrames, firstChunkFrames < 1 {
            throw Qwen3TTSVoiceError.invalidSchedule(
                "firstChunkFrames must be >= 1, got \(firstChunkFrames)"
            )
        }
        if let maxChunkFrames, maxChunkFrames < 1 {
            throw Qwen3TTSVoiceError.invalidSchedule(
                "maxChunkFrames must be >= 1, got \(maxChunkFrames)"
            )
        }
        if let firstChunkFrames, let maxChunkFrames, maxChunkFrames < firstChunkFrames {
            throw Qwen3TTSVoiceError.invalidSchedule(
                "maxChunkFrames \(maxChunkFrames) < firstChunkFrames \(firstChunkFrames)"
            )
        }
        if let maxFrames, maxFrames < 1 {
            throw Qwen3TTSVoiceError.invalidSchedule(
                "maxFrames must be >= 1, got \(maxFrames)"
            )
        }
    }

    /// The package-authored voice defaults, or nil when none are present.
    /// A malformed file fails loudly. `SMELT_NO_VOICE_DEFAULTS=1` selects
    /// built-in defaults; `env` is injectable for tests.
    public static func load(
        packagePath: String,
        env: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> Qwen3TTSVoice? {
        guard env["SMELT_NO_VOICE_DEFAULTS"] != "1" else { return nil }
        let url = URL(fileURLWithPath: packagePath).appendingPathComponent(fileName)
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let voice = try JSONDecoder().decode(Qwen3TTSVoice.self, from: Data(contentsOf: url))
        try voice.validate()
        return voice
    }

    public func write(packagePath: String) throws {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        let url = URL(fileURLWithPath: packagePath).appendingPathComponent(Self.fileName)
        try enc.encode(self).write(to: url)
    }
}
