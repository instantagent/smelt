#if canImport(WhisperKit)
import CoreML
import Foundation
import WhisperKit

public final class WhisperTinyDecodingInputs: DecodingInputsType {
    public var initialPrompt: [Int]
    public var inputIds: MLMultiArray
    public var cacheLength: MLMultiArray

    public init(initialPrompt: [Int]) throws {
        self.initialPrompt = initialPrompt
        self.inputIds = try MLMultiArray(shape: [1], dataType: .int32, initialValue: Int32(0))
        self.cacheLength = try MLMultiArray(shape: [1], dataType: .int32, initialValue: Int32(0))
    }

    public func reset(prefilledCacheSize: Int, maxTokenContext: Int) {
        cacheLength[0] = NSNumber(value: prefilledCacheSize)
        inputIds[0] = 0
    }
}

public final class WhisperTinyWhisperKitDecoder: TextDecoding {
    public var tokenizer: WhisperTokenizer?
    public var prefillData: WhisperMLModel?
    public var isModelMultilingual: Bool = true
    public var logitsFilters: [any LogitsFiltering]?

    private let runtime: WhisperTinyCPUDecoder

    public init(
        runtime: WhisperTinyCPUDecoder,
        tokenizer: WhisperTokenizer? = nil,
        logitsFilters: [any LogitsFiltering]? = []
    ) {
        self.runtime = runtime
        self.tokenizer = tokenizer
        self.logitsFilters = logitsFilters
    }

    public convenience init(
        weightsPath: String? = nil,
        tokenizer: WhisperTokenizer? = nil,
        logitsFilters: [any LogitsFiltering]? = []
    ) throws {
        try self.init(
            runtime: WhisperTinyCPUDecoder(weightsPath: weightsPath),
            tokenizer: tokenizer,
            logitsFilters: logitsFilters
        )
    }

    public var supportsWordTimestamps: Bool { true }
    public var logitsSize: Int? { WhisperTinyConstants.vocabSize }
    public var kvCacheEmbedDim: Int? { WhisperTinyConstants.layerCount * WhisperTinyConstants.dModel }
    public var kvCacheMaxSequenceLength: Int? { WhisperTinyConstants.maxTargetPositions }
    public var windowSize: Int? { WhisperTinyConstants.sourceLength }
    public var embedSize: Int? { WhisperTinyConstants.dModel }

    public func predictLogits(
        _ inputs: any TextDecoderInputType
    ) async throws -> TextDecoderOutputType? {
        throw WhisperError.decodingLogitsFailed(
            "WhisperTinyWhisperKitDecoder does not support WhisperKit's CoreML cache tensors. Use decodeText/detectLanguage with WhisperTinyDecodingInputs."
        )
    }

    public func prepareDecoderInputs(withPrompt initialPrompt: [Int]) throws -> any DecodingInputsType {
        try WhisperTinyDecodingInputs(initialPrompt: initialPrompt)
    }

    public func prefillDecoderInputs(
        _ decoderInputs: any DecodingInputsType,
        withOptions options: DecodingOptions?
    ) async throws -> any DecodingInputsType {
        guard let tokenizer else {
            throw WhisperError.tokenizerUnavailable()
        }
        guard let decoderInputs = decoderInputs as? WhisperTinyDecodingInputs else {
            throw WhisperError.prepareDecoderInputsFailed("Expected WhisperTinyDecodingInputs")
        }

        var prefillTokens = [tokenizer.specialTokens.startOfTranscriptToken]
        var taskToken = tokenizer.specialTokens.transcribeToken
        var languageToken = tokenizer.specialTokens.englishToken

        if let options {
            if isModelMultilingual {
                let languageTokenString = "<|\(options.language ?? Constants.defaultLanguageCode)|>"
                languageToken = tokenizer.convertTokenToId(languageTokenString) ?? tokenizer.specialTokens.englishToken
                prefillTokens.append(languageToken)

                let taskTokenString = "<|\(options.task)|>"
                taskToken = tokenizer.convertTokenToId(taskTokenString) ?? tokenizer.specialTokens.transcribeToken
                prefillTokens.append(taskToken)
            }

            let timestampsToken =
                options.withoutTimestamps
                ? tokenizer.specialTokens.noTimestampsToken
                : tokenizer.specialTokens.timeTokenBegin
            prefillTokens.append(timestampsToken)

            if let promptTokens = options.promptTokens {
                let maxPromptLen = (Constants.maxTokenContext / 2) - 1
                let trimmedPromptTokens = Array(promptTokens.suffix(maxPromptLen))
                    .filter { $0 < tokenizer.specialTokens.specialTokenBegin }
                prefillTokens = [tokenizer.specialTokens.startOfPreviousToken] + trimmedPromptTokens + prefillTokens
            }

            if let prefixTokens = options.prefixTokens {
                let trimmedPrefixTokens = Array(prefixTokens.suffix(Constants.maxTokenContext / 2))
                    .filter { $0 < tokenizer.specialTokens.specialTokenBegin }
                prefillTokens.append(contentsOf: trimmedPrefixTokens)
            }
        }

        decoderInputs.initialPrompt = prefillTokens
        decoderInputs.cacheLength[0] = 0
        decoderInputs.inputIds[0] = NSNumber(value: taskToken)
        return decoderInputs
    }

    public func prefillKVCache(
        withTask task: MLMultiArray,
        andLanguage language: MLMultiArray
    ) async throws -> DecodingCache? {
        nil
    }

    public func detectLanguage(
        from encoderOutput: any AudioEncoderOutputType,
        using decoderInputs: any DecodingInputsType,
        sampler tokenSampler: TokenSampling,
        options: DecodingOptions,
        temperature: FloatType
    ) async throws -> DecodingResult {
        guard let tokenizer else {
            throw WhisperError.tokenizerUnavailable()
        }
        guard let logitsSize else {
            throw WhisperError.modelsUnavailable("Missing logits size")
        }

        let encoderOutput = try encoderOutput.asFloatArray()
        var state = runtime.makeState(encoderOutput: encoderOutput)
        let startOfTranscript = tokenizer.specialTokens.startOfTranscriptToken
        let step = runtime.decodeStep(token: startOfTranscript, state: &state)
        var logits = try step.logits.asLogitsMLMultiArray()

        let filter = LanguageLogitsFilter(
            allLanguageTokens: tokenizer.allLanguageTokens,
            logitsDim: logitsSize,
            sampleBegin: 0
        )
        logits = filter.filterLogits(logits, withTokens: [startOfTranscript])

        let sample = tokenSampler.update(tokens: [startOfTranscript], logits: logits, logProbs: [0])
        let sampledToken = sample.tokens.last ?? tokenizer.specialTokens.englishToken
        let sampledLanguage = tokenizer.decode(tokens: [sampledToken]).trimmingSpecialTokenCharacters()
        let language = Constants.languageCodes.contains(sampledLanguage)
            ? sampledLanguage
            : Constants.defaultLanguageCode

        return DecodingResult(
            language: language,
            languageProbs: [language: sample.logProbs.last ?? 0],
            tokens: [],
            tokenLogProbs: [],
            text: "",
            avgLogProb: 0,
            noSpeechProb: 0,
            temperature: Float(temperature),
            compressionRatio: 0,
            cache: nil,
            timings: TranscriptionTimings(),
            fallback: nil
        )
    }

    public func decodeText(
        from encoderOutput: any AudioEncoderOutputType,
        using decoderInputs: any DecodingInputsType,
        sampler tokenSampler: TokenSampling,
        options: DecodingOptions,
        callback: ((TranscriptionProgress) -> Bool?)?
    ) async throws -> DecodingResult {
        guard let tokenizer else {
            throw WhisperError.tokenizerUnavailable()
        }
        guard let decoderInputs = decoderInputs as? WhisperTinyDecodingInputs else {
            throw WhisperError.prepareDecoderInputsFailed("Expected WhisperTinyDecodingInputs")
        }

        let encoderOutput = try encoderOutput.asFloatArray()
        var state = runtime.makeState(encoderOutput: encoderOutput)
        var currentTokens = decoderInputs.initialPrompt
        var logProbs = Array(repeating: Float(0), count: currentTokens.count)
        var alignmentRows: [[Float]] = []
        var timings = TranscriptionTimings()

        let prefilledIndex = decoderInputs.cacheLength[0].intValue
        let initialPromptIndex = decoderInputs.initialPrompt.count
        let loopCount = min(
            options.sampleLength,
            Constants.maxTokenContext - 1,
            WhisperTinyConstants.maxTargetPositions - 1
        )
        let filters = createLogitsFilters(
            options: options,
            prefilledIndex: prefilledIndex,
            initialPromptIndex: initialPromptIndex,
            tokenizer: tokenizer
        )

        var isFirstTokenLogProbTooLow = false

        for tokenIndex in prefilledIndex..<loopCount {
            let loopStart = Date()
            let isPrefill = tokenIndex < initialPromptIndex - 1
            let isFirstToken = tokenIndex == prefilledIndex

            guard tokenIndex < currentTokens.count else {
                break
            }

            let token = currentTokens[tokenIndex]
            decoderInputs.inputIds[0] = NSNumber(value: token)
            decoderInputs.cacheLength[0] = NSNumber(value: tokenIndex)

            let inferenceStart = Date()
            let step = runtime.decodeStep(token: token, state: &state)
            timings.decodingPredictions += Date().timeIntervalSince(inferenceStart)
            alignmentRows.append(step.alignmentRow)

            let filteringStart = Date()
            var logits = try step.logits.asLogitsMLMultiArray()
            for filter in filters {
                logits = filter.filterLogits(logits, withTokens: currentTokens)
            }
            timings.decodingFiltering += Date().timeIntervalSince(filteringStart)

            let samplingStart = Date()
            let sampled = tokenSampler.update(tokens: currentTokens, logits: logits, logProbs: logProbs)
            timings.decodingSampling += Date().timeIntervalSince(samplingStart)

            let nextToken = sampled.tokens.last ?? tokenizer.specialTokens.endToken
            let nextLogProb = sampled.logProbs.last ?? 0
            isFirstTokenLogProbTooLow =
                if isFirstToken,
                   let threshold = options.firstTokenLogProbThreshold,
                   nextLogProb < threshold {
                    true
                } else {
                    false
                }

            let isSegmentCompleted =
                sampled.completed ||
                currentTokens.count >= Constants.maxTokenContext - 1 ||
                isFirstTokenLogProbTooLow

            if isSegmentCompleted {
                timings.decodingLoop += Date().timeIntervalSince(loopStart)
                timings.totalDecodingLoops += 1
                break
            }

            if !isPrefill {
                currentTokens.append(nextToken)
                logProbs.append(nextLogProb)

                if let callback {
                    let averageLogProb = logProbs.reduce(0, +) / Float(logProbs.count)
                    let currentTranscript = tokenizer.decode(tokens: currentTokens)
                    let compressionRatio = TextUtilities.compressionRatio(of: currentTokens)
                    _ = callback(
                        TranscriptionProgress(
                            timings: timings,
                            text: currentTranscript,
                            tokens: currentTokens,
                            avgLogprob: averageLogProb,
                            compressionRatio: compressionRatio
                        )
                    )
                }
            }

            timings.decodingLoop += Date().timeIntervalSince(loopStart)
            timings.totalDecodingLoops += 1
        }

        let finalSampling = tokenSampler.finalize(tokens: currentTokens, logProbs: logProbs)
        let segmentTokens = finalSampling.tokens
        let segmentLogProbs = finalSampling.logProbs

        let startIndex = segmentTokens.firstIndex(of: tokenizer.specialTokens.startOfTranscriptToken) ?? 0
        let endIndex = segmentTokens.firstIndex(of: tokenizer.specialTokens.endToken) ?? segmentTokens.count - 1
        let filteredTokens = Array(segmentTokens[startIndex...endIndex])
        let filteredLogProbs = Array(segmentLogProbs[startIndex...endIndex])

        var tokenLogProbs = [[Int: Float]]()
        for (token, logProb) in zip(filteredTokens, filteredLogProbs) {
            tokenLogProbs.append([token: logProb])
        }

        let averageLogProb = filteredLogProbs.isEmpty ? 0 : filteredLogProbs.reduce(0, +) / Float(filteredLogProbs.count)
        let wordTokens = filteredTokens.filter { $0 < tokenizer.specialTokens.specialTokenBegin }
        let compressionRatio = TextUtilities.compressionRatio(of: wordTokens)
        let transcript = tokenizer.decode(tokens: filteredTokens)
        let alignmentWeights = try alignmentRows.asAlignmentMLMultiArray(sourceLength: WhisperTinyConstants.sourceLength)

        let language = options.language ?? Constants.defaultLanguageCode
        let fallback = DecodingFallback(
            options: options,
            isFirstTokenLogProbTooLow: isFirstTokenLogProbTooLow,
            noSpeechProb: 0,
            compressionRatio: compressionRatio,
            avgLogProb: averageLogProb
        )

        return DecodingResult(
            language: language,
            languageProbs: [language: 0],
            tokens: filteredTokens,
            tokenLogProbs: tokenLogProbs,
            text: transcript,
            avgLogProb: averageLogProb,
            noSpeechProb: 0,
            temperature: options.temperature,
            compressionRatio: compressionRatio,
            cache: DecodingCache(
                keyCache: nil,
                valueCache: nil,
                alignmentWeights: alignmentWeights
            ),
            timings: timings,
            fallback: fallback
        )
    }

    public static func updateKVCache(
        keyTensor: MLMultiArray,
        keySlice: MLMultiArray,
        valueTensor: MLMultiArray,
        valueSlice: MLMultiArray,
        insertAtIndex index: Int
    ) {}

    private func createLogitsFilters(
        options: DecodingOptions,
        prefilledIndex: Int,
        initialPromptIndex: Int,
        tokenizer: WhisperTokenizer
    ) -> [any LogitsFiltering] {
        var filters: [any LogitsFiltering] = logitsFilters ?? []

        if options.suppressBlank {
            filters.append(
                SuppressBlankFilter(
                    specialTokens: tokenizer.specialTokens,
                    sampleBegin: prefilledIndex
                )
            )
        }

        if !options.supressTokens.isEmpty {
            let filteredTokens = options.supressTokens.filter { $0 < tokenizer.specialTokens.specialTokenBegin }
            filters.append(SuppressTokensFilter(suppressTokens: filteredTokens))
        }

        if !options.withoutTimestamps {
            let maxInitialTimestampIndex: Int?
            if let maxInitialTimestamp = options.maxInitialTimestamp {
                maxInitialTimestampIndex = Int(maxInitialTimestamp / WhisperKit.secondsPerTimeToken)
            } else {
                maxInitialTimestampIndex = nil
            }
            filters.append(
                TimestampRulesFilter(
                    specialTokens: tokenizer.specialTokens,
                    sampleBegin: initialPromptIndex,
                    maxInitialTimestampIndex: maxInitialTimestampIndex,
                    isModelMultilingual: isModelMultilingual
                )
            )
        }

        return filters
    }
}

private extension AudioEncoderOutputType {
    func asFloatArray() throws -> [Float] {
        guard let multiArray = self as? MLMultiArray else {
            throw WhisperError.prepareDecoderInputsFailed("AudioEncoderOutputType must be MLMultiArray")
        }

        switch multiArray.dataType {
        case .float16:
            let values = UnsafeMutablePointer<Float16>(OpaquePointer(multiArray.dataPointer))
            return (0..<multiArray.count).map { Float(values[$0]) }
        case .float32:
            let values = UnsafeMutablePointer<Float>(OpaquePointer(multiArray.dataPointer))
            return (0..<multiArray.count).map { values[$0] }
        default:
            throw WhisperError.prepareDecoderInputsFailed("Unsupported encoder output dtype")
        }
    }
}

private extension Array where Element == Float {
    func asLogitsMLMultiArray() throws -> MLMultiArray {
        let logits = try MLMultiArray(
            shape: [1, 1, NSNumber(value: count)],
            dataType: .float16,
            initialValue: FloatType(0)
        )

        logits.withUnsafeMutableBufferPointer(ofType: FloatType.self) { pointer, strides in
            for (index, value) in enumerated() {
                pointer[index * strides[2]] = FloatType(value)
            }
        }

        return logits
    }
}

private extension Array where Element == [Float] {
    func asAlignmentMLMultiArray(sourceLength: Int) throws -> MLMultiArray {
        let rowCount = count
        let alignment = try MLMultiArray(
            shape: [NSNumber(value: rowCount), NSNumber(value: sourceLength)],
            dataType: .float16,
            initialValue: FloatType(0)
        )

        alignment.withUnsafeMutableBufferPointer(ofType: FloatType.self) { pointer, strides in
            for row in 0..<rowCount {
                for column in 0..<sourceLength {
                    pointer[row * strides[0] + column * strides[1]] = FloatType(self[row][column])
                }
            }
        }

        return alignment
    }
}
#endif
