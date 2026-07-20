import XCTest
@testable import SmeltRuntime
import SmeltSchema

final class QwenSmokeTests: XCTestCase {

    private func resolvePackagePath() throws -> String {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["SMELT_RUN_TEXT_SMOKE"] == "1",
            "Qwen smoke test is opt-in"
        )

        let fm = FileManager.default
        let env = ProcessInfo.processInfo.environment
        if let explicit = env["SMELT_TEXT_PACKAGE"],
           fm.fileExists(atPath: "\(explicit)/manifest.json")
        {
            return explicit
        }

        let canonical = fm.currentDirectoryPath
            + "/artifacts/qwen35-2b-qmm16x128/Qwen_Qwen3.5-2B.smeltpkg"
        try XCTSkipUnless(
            fm.fileExists(atPath: "\(canonical)/manifest.json"),
            "Missing canonical Qwen package at \(canonical). Build it with bash tools/build-qwen35-2b.sh"
        )
        return canonical
    }

    private func resolveQwen0808PackagePath() throws -> String {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["SMELT_RUN_TEXT_SMOKE"] == "1",
            "Qwen smoke test is opt-in"
        )

        let fm = FileManager.default
        let env = ProcessInfo.processInfo.environment
        if let explicit = env["SMELT_QWEN_0_8B_PACKAGE"],
           fm.fileExists(atPath: "\(explicit)/manifest.json")
        {
            return explicit
        }

        let canonical = fm.currentDirectoryPath
            + "/artifacts/qwen35-0.8b-qmm16x128/Qwen_Qwen3.5-0.8B.smeltpkg"
        try XCTSkipUnless(
            fm.fileExists(atPath: "\(canonical)/manifest.json"),
            "Missing canonical Qwen 0.8B package at \(canonical). Build it with bash tools/build-qwen35-0.8b.sh"
        )
        return canonical
    }

    private func resolveQwen4BPackagePath() throws -> String {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["SMELT_RUN_TEXT_SMOKE"] == "1",
            "Qwen smoke test is opt-in"
        )

        let fm = FileManager.default
        let env = ProcessInfo.processInfo.environment
        if let explicit = env["SMELT_QWEN_4B_PACKAGE"],
           fm.fileExists(atPath: "\(explicit)/manifest.json")
        {
            return explicit
        }

        let canonical = fm.currentDirectoryPath
            + "/artifacts/qwen35-4b-qmm16x128/Qwen_Qwen3.5-4B.smeltpkg"
        try XCTSkipUnless(
            fm.fileExists(atPath: "\(canonical)/manifest.json"),
            "Missing canonical Qwen 4B package at \(canonical). Build it with bash tools/build-qwen35-4b.sh"
        )
        return canonical
    }

    private func qwenChatTokenIds(
        prompt: String,
        tokenizer: SmeltTokenizer
    ) throws -> [Int32] {
        guard let imStart = tokenizer.addedTokenId(for: "<|im_start|>"),
              let imEnd = tokenizer.addedTokenId(for: "<|im_end|>"),
              let think = tokenizer.addedTokenId(for: "<think>"),
              let thinkEnd = tokenizer.addedTokenId(for: "</think>")
        else {
            throw XCTSkip("Tokenizer is missing Qwen chat special tokens")
        }

        var ids: [Int32] = [Int32(imStart)]
        ids += tokenizer.encode("user")
        ids += tokenizer.encode("\n")
        ids += tokenizer.encode(prompt)
        ids += [Int32(imEnd)]
        ids += tokenizer.encode("\n")
        ids += [Int32(imStart)]
        ids += tokenizer.encode("assistant")
        ids += tokenizer.encode("\n")
        ids += [Int32(think)]
        ids += tokenizer.encode("\n\n")
        ids += [Int32(thinkEnd)]
        ids += tokenizer.encode("\n\n")
        return ids
    }

    private func makePrompt(
        tokenizer: SmeltTokenizer,
        targetMin: Int,
        targetMax: Int
    ) throws -> (String, [Int32]) {
        let chunk =
            "Summarize how Paris shaped European history, art, science, and diplomacy in two clear sentences. "
        var prompt = chunk
        var ids = try qwenChatTokenIds(prompt: prompt, tokenizer: tokenizer)
        while ids.count < targetMin {
            let grown = prompt + chunk
            let grownIds = try qwenChatTokenIds(prompt: grown, tokenizer: tokenizer)
            if grownIds.count > targetMax { break }
            prompt = grown
            ids = grownIds
        }

        for filler in [" Paris", " history", " art", " science", ".", "!"] {
            if ids.count >= targetMin { break }
            let grown = prompt + filler
            let grownIds = try qwenChatTokenIds(prompt: grown, tokenizer: tokenizer)
            if grownIds.count <= targetMax {
                prompt = grown
                ids = grownIds
            }
        }

        while ids.count < targetMin {
            let grown = prompt + " Paris"
            let grownIds = try qwenChatTokenIds(prompt: grown, tokenizer: tokenizer)
            if grownIds.count > targetMax { break }
            prompt = grown
            ids = grownIds
        }

        XCTAssertGreaterThanOrEqual(ids.count, targetMin, "Could not build a prompt near the prefill boundary")
        XCTAssertLessThanOrEqual(ids.count, targetMax, "Prompt exceeded the package prefill boundary")
        return (prompt, ids)
    }

    private func makeBoundaryPrompt(tokenizer: SmeltTokenizer) throws -> (String, [Int32]) {
        try makePrompt(tokenizer: tokenizer, targetMin: 255, targetMax: 255)
    }

    private func decodeReferenceTokens(
        packagePath: String,
        promptTokenIds: [Int32],
        maxTokens: Int
    ) throws -> [Int32] {
        let runtime = try SmeltRuntime(packagePath: packagePath)
        var cur: Int32 = 0
        for (i, tid) in promptTokenIds.enumerated() {
            cur = try runtime.decodeStep(tokenId: tid, position: Int32(i))
        }
        guard maxTokens > 0 else { return [] }

        var tokens: [Int32] = [cur]
        var pos = promptTokenIds.count
        while tokens.count < maxTokens {
            cur = try runtime.decodeStep(tokenId: cur, position: Int32(pos))
            tokens.append(cur)
            pos += 1
        }
        return tokens
    }

    private func restoredDecodeTokens(
        packagePath: String,
        snapshot: SmeltPromptSnapshot,
        suffixTokenIds: [Int32],
        maxTokens: Int
    ) throws -> [Int32] {
        let runtime = try SmeltRuntime(packagePath: packagePath)
        try runtime.prepareForRequest(
            batchCapacity: max(snapshot.replayTokenIds.count + suffixTokenIds.count, 1),
            contextCapacity: max(snapshot.promptLength + suffixTokenIds.count, 1)
        )
        runtime.resetWorkingBuffers()
        try runtime.restorePromptSnapshot(snapshot)

        var cur: Int32 = snapshot.nextToken
        var pos = snapshot.capturedLength
        for tid in snapshot.replayTokenIds + suffixTokenIds {
            cur = try runtime.decodeStep(tokenId: tid, position: Int32(pos))
            pos += 1
        }

        guard maxTokens > 0 else { return [] }
        var tokens: [Int32] = [cur]
        while tokens.count < maxTokens {
            cur = try runtime.decodeStep(tokenId: cur, position: Int32(pos))
            tokens.append(cur)
            pos += 1
        }
        return tokens
    }

    private func loadTraceMarkers(packagePath: String) throws -> SmeltTraceMarkers {
        let path = "\(packagePath)/trace_markers.json"
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        return try JSONDecoder().decode(SmeltTraceMarkers.self, from: data)
    }

    private func loadManifest(packagePath: String) throws -> SmeltManifest {
        let path = "\(packagePath)/manifest.json"
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        return try JSONDecoder().decode(SmeltManifest.self, from: data)
    }

    private func captureDecodeMarkerValues(
        packagePath: String,
        tokenIds: [Int32],
        marker: SmeltTraceMarker,
        count: Int
    ) throws -> (token: Int32, values: [Float]) {
        try captureDecodeSlotValues(
            packagePath: packagePath,
            tokenIds: tokenIds,
            maxDispatches: marker.dispatchCount,
            slot: marker.bufferSlot,
            count: count
        )
    }

    private func captureDecodeSlotValues(
        packagePath: String,
        tokenIds: [Int32],
        maxDispatches: Int,
        slot: Int,
        count: Int
    ) throws -> (token: Int32, values: [Float]) {
        let runtime = try SmeltRuntime(packagePath: packagePath)
        runtime.resetWorkingBuffers()
        if tokenIds.count > 1 {
            for position in 0..<(tokenIds.count - 1) {
                _ = try runtime.decodeStep(
                    tokenId: tokenIds[position],
                    position: Int32(position)
                )
            }
        }
        let token = try runtime.debugDecodeStep(
            tokenId: tokenIds[tokenIds.count - 1],
            position: Int32(tokenIds.count - 1),
            maxDispatches: maxDispatches
        )
        return (
            token,
            runtime.dumpSlot(slot, count: count)
        )
    }

    private func referenceAffineOutput(
        weightsData: Data,
        entry: SmeltWeightEntry,
        input: [Float]
    ) throws -> [Float] {
        XCTAssertEqual(entry.dtype, .affineU4)
        let rows = try XCTUnwrap(entry.shape.first)
        let cols = try XCTUnwrap(entry.shape.dropFirst().first)
        let groupSize = try XCTUnwrap(entry.groupSize)
        let packedRowStride = try XCTUnwrap(entry.packedRowStride)
        let paddedCols = entry.paddedCols ?? cols
        let scalesOffset = Int(try XCTUnwrap(entry.scalesOffset))
        let biasesOffset = Int(try XCTUnwrap(entry.biasesOffset))

        XCTAssertEqual(input.count, cols)
        XCTAssertEqual(paddedCols % groupSize, 0)

        let groupsPerRow = paddedCols / groupSize
        let scaleStart = scalesOffset / MemoryLayout<Float16>.stride
        let scaleEnd = scaleStart + rows * groupsPerRow
        let biasStart = biasesOffset / MemoryLayout<Float16>.stride
        let biasEnd = biasStart + rows * groupsPerRow
        let scales = weightsData.withUnsafeBytes {
            let values = $0.bindMemory(to: Float16.self)
            return Array(values[scaleStart..<scaleEnd])
        }
        let biases = weightsData.withUnsafeBytes {
            let values = $0.bindMemory(to: Float16.self)
            return Array(values[biasStart..<biasEnd])
        }

        var output = [Float](repeating: 0, count: rows)
        weightsData.withUnsafeBytes { raw in
            let bytes = raw.bindMemory(to: UInt8.self)
            for row in 0..<rows {
                let rowBase = Int(entry.offset) + row * packedRowStride
                let sbBase = row * groupsPerRow
                var acc: Float = 0
                for group in 0..<groupsPerRow {
                    let scale = Float(scales[sbBase + group])
                    let bias = Float(biases[sbBase + group])
                    let colBase = group * groupSize
                    var dot: Float = 0
                    var xsum: Float = 0
                    for i in 0..<groupSize {
                        let col = colBase + i
                        let byte = bytes[rowBase + (col / 2)]
                        let nibble = (col & 1) == 0 ? (byte & 0x0F) : (byte >> 4)
                        let x = input[col]
                        dot += Float(nibble) * x
                        xsum += x
                    }
                    acc += scale * dot + bias * xsum
                }
                output[row] = Float(Float16(acc))
            }
        }
        return output
    }

    private func batchedRowStrideElements(
        for slot: Int,
        manifest: SmeltManifest
    ) -> Int {
        let compiledMaxBatch = manifest.prefill?.maxBatchSize ?? 0
        guard let slotInfo = manifest.buffers.slots.first(where: { $0.index == slot }),
              compiledMaxBatch > 0
        else {
            return manifest.config.hiddenSize
        }

        guard let bytesPerElement = slotInfo.dtype.bytesPerElement else {
            return manifest.config.hiddenSize
        }

        let totalElements = slotInfo.sizeBytes / bytesPerElement
        guard totalElements > 0, totalElements % compiledMaxBatch == 0 else {
            return manifest.config.hiddenSize
        }
        return totalElements / compiledMaxBatch
    }

    private func continuationTraceFirstMismatch(
        packagePath: String,
        prefixTokenIds: [Int32],
        continuationTokenIds: [Int32]
    ) throws -> String? {
        let trace = try loadTraceMarkers(packagePath: packagePath)
        let manifest = try loadManifest(packagePath: packagePath)
        let decodeByLabel = Dictionary(uniqueKeysWithValues: trace.decode.map { ($0.label, $0) })
        let sharedMarkers = trace.prefill.filter { decodeByLabel[$0.label] != nil }
        guard !sharedMarkers.isEmpty else { return nil }

        var recentDiffs: [(String, Float)] = []
        for marker in sharedMarkers {
            let rowStride = batchedRowStrideElements(
                for: marker.bufferSlot,
                manifest: manifest
            )
            let prefillRuntime = try SmeltRuntime(packagePath: packagePath)
            try prefillRuntime.prepareForRequest(
                batchCapacity: max(prefixTokenIds.count, continuationTokenIds.count, 1),
                contextCapacity: max(prefixTokenIds.count + continuationTokenIds.count, 1)
            )
            prefillRuntime.resetWorkingBuffers()
            _ = try prefillRuntime.prefillStep(tokenIds: prefixTokenIds, startPos: 0)
            _ = try prefillRuntime.debugPrefillStep(
                tokenIds: continuationTokenIds,
                startPos: Int32(prefixTokenIds.count),
                maxDispatches: marker.dispatchCount
            )
            let prefillValues = prefillRuntime.dumpSlot(
                marker.bufferSlot,
                elementOffset: (continuationTokenIds.count - 1) * rowStride,
                count: rowStride
            )

            let decodeMarker = decodeByLabel[marker.label]!
            let decodeRuntime = try SmeltRuntime(packagePath: packagePath)
            decodeRuntime.resetWorkingBuffers()
            if !prefixTokenIds.isEmpty {
                for (i, tid) in prefixTokenIds.enumerated() {
                    _ = try decodeRuntime.decodeStep(tokenId: tid, position: Int32(i))
                }
            }
            if continuationTokenIds.count > 1 {
                for (i, tid) in continuationTokenIds.dropLast().enumerated() {
                    _ = try decodeRuntime.decodeStep(
                        tokenId: tid,
                        position: Int32(prefixTokenIds.count + i)
                    )
                }
            }
            _ = try decodeRuntime.debugDecodeStep(
                tokenId: continuationTokenIds[continuationTokenIds.count - 1],
                position: Int32(prefixTokenIds.count + continuationTokenIds.count - 1),
                maxDispatches: decodeMarker.dispatchCount
            )
            let decodeValues = decodeRuntime.dumpSlot(
                decodeMarker.bufferSlot,
                count: rowStride
            )

            var maxDiff: Float = 0
            for idx in 0..<min(prefillValues.count, decodeValues.count) {
                maxDiff = max(maxDiff, abs(prefillValues[idx] - decodeValues[idx]))
            }

            recentDiffs.append((marker.label, maxDiff))
            if recentDiffs.count > 4 {
                recentDiffs.removeFirst(recentDiffs.count - 4)
            }

            if maxDiff > 0.0001 {
                let context = recentDiffs.map { "\($0.0)=\($0.1)" }.joined(separator: ", ")
                return "\(marker.label) diff=\(maxDiff) [\(context)]"
            }
        }

        return nil
    }

    private func fullBatchMarkerDiff(
        packagePath: String,
        prefixTokenIds: [Int32],
        continuationTokenIds: [Int32],
        label: String
    ) throws -> String? {
        let trace = try loadTraceMarkers(packagePath: packagePath)
        let manifest = try loadManifest(packagePath: packagePath)

        guard let prefillMarker = trace.prefill.first(where: { $0.label == label }),
              let decodeMarker = trace.decode.first(where: { $0.label == label })
        else {
            return nil
        }

        let rowStride = batchedRowStrideElements(
            for: prefillMarker.bufferSlot,
            manifest: manifest
        )

        let prefillRuntime = try SmeltRuntime(packagePath: packagePath)
        try prefillRuntime.prepareForRequest(
            batchCapacity: max(prefixTokenIds.count, continuationTokenIds.count, 1),
            contextCapacity: max(prefixTokenIds.count + continuationTokenIds.count, 1)
        )
        prefillRuntime.resetWorkingBuffers()
        _ = try prefillRuntime.prefillStep(tokenIds: prefixTokenIds, startPos: 0)
        _ = try prefillRuntime.debugPrefillStep(
            tokenIds: continuationTokenIds,
            startPos: Int32(prefixTokenIds.count),
            maxDispatches: prefillMarker.dispatchCount
        )
        let prefillValues = prefillRuntime.dumpSlot(
            prefillMarker.bufferSlot,
            count: continuationTokenIds.count * rowStride
        )

        var decodeValues: [Float] = []
        decodeValues.reserveCapacity(continuationTokenIds.count * rowStride)
        var worstPos = -1
        var worstIdx = -1
        var worstDiff: Float = 0

        for pos in continuationTokenIds.indices {
            let decodeRuntime = try SmeltRuntime(packagePath: packagePath)
            decodeRuntime.resetWorkingBuffers()
            for (i, tid) in prefixTokenIds.enumerated() {
                _ = try decodeRuntime.decodeStep(tokenId: tid, position: Int32(i))
            }
            if pos > 0 {
                for (i, tid) in continuationTokenIds[..<pos].enumerated() {
                    _ = try decodeRuntime.decodeStep(
                        tokenId: tid,
                        position: Int32(prefixTokenIds.count + i)
                    )
                }
            }
            _ = try decodeRuntime.debugDecodeStep(
                tokenId: continuationTokenIds[pos],
                position: Int32(prefixTokenIds.count + pos),
                maxDispatches: decodeMarker.dispatchCount
            )
            let row = decodeRuntime.dumpSlot(
                decodeMarker.bufferSlot,
                count: rowStride
            )
            decodeValues += row

            let prefillBase = pos * rowStride
            for idx in 0..<rowStride {
                let diff = abs(prefillValues[prefillBase + idx] - row[idx])
                if diff > worstDiff {
                    worstDiff = diff
                    worstPos = pos
                    worstIdx = idx
                }
            }
        }

        guard worstDiff > 0.0001 else { return "\(label)=0.0" }
        return "\(label)=\(worstDiff) at pos=\(worstPos) elem=\(worstIdx)"
    }

    private func continuedPrefillTokens(
        packagePath: String,
        capturedTokenIds: [Int32],
        continuationTokenIds: [Int32],
        maxTokens: Int
    ) throws -> [Int32] {
        let runtime = try SmeltRuntime(packagePath: packagePath)
        try runtime.prepareForRequest(
            batchCapacity: max(continuationTokenIds.count, capturedTokenIds.count, 1),
            contextCapacity: max(capturedTokenIds.count + continuationTokenIds.count, 1)
        )
        runtime.resetWorkingBuffers()

        let capturedCur = try runtime.prefillStep(tokenIds: capturedTokenIds, startPos: 0)
        let cur: Int32
        let pos: Int
        if continuationTokenIds.isEmpty {
            cur = capturedCur
            pos = capturedTokenIds.count
        } else {
            cur = try runtime.prefillStep(
                tokenIds: continuationTokenIds,
                startPos: Int32(capturedTokenIds.count)
            )
            pos = capturedTokenIds.count + continuationTokenIds.count
        }

        guard maxTokens > 0 else { return [] }
        var tokens: [Int32] = [cur]
        var decodeCur = cur
        var decodePos = pos
        while tokens.count < maxTokens {
            decodeCur = try runtime.decodeStep(tokenId: decodeCur, position: Int32(decodePos))
            tokens.append(decodeCur)
            decodePos += 1
        }
        return tokens
    }

    func testLongPromptRepeatedFreshRuntimeStable() throws {
        let packagePath = try resolvePackagePath()
        let tokenizer = try SmeltTokenizer(path: "\(packagePath)/tokenizer.json")
        let (prompt, inputIds) = try makeBoundaryPrompt(tokenizer: tokenizer)

        XCTAssertFalse(prompt.isEmpty)
        XCTAssertGreaterThanOrEqual(
            inputIds.count,
            255,
            "Smoke prompt must reach the dynamic prefill boundary"
        )
        XCTAssertLessThanOrEqual(
            inputIds.count,
            255,
            "Smoke prompt must stay within the package sequence limit"
        )

        let decodePos = Int32(inputIds.count)

        let firstRuntime = try SmeltRuntime(packagePath: packagePath)
        let firstToken = try firstRuntime.prefillStep(tokenIds: inputIds, startPos: 0)
        let firstFollowup = try firstRuntime.decodeStep(tokenId: firstToken, position: decodePos)

        let secondRuntime = try SmeltRuntime(packagePath: packagePath)
        let secondToken = try secondRuntime.prefillStep(tokenIds: inputIds, startPos: 0)
        let secondFollowup = try secondRuntime.decodeStep(tokenId: secondToken, position: decodePos)

        XCTAssertEqual(
            [firstToken, firstFollowup],
            [secondToken, secondFollowup],
            "Fresh-runtime prefill/decode diverged on the repeated boundary prompt"
        )

        let decoded = tokenizer.decode([firstToken, firstFollowup])
        XCTAssertFalse(decoded.isEmpty, "Smoke run produced empty decoded output")
    }

    func testQwenTemperatureSamplingGPUIsAvailableAndDeterministic() throws {
        let packagePath = try resolvePackagePath()
        let firstRuntime = try SmeltRuntime(packagePath: packagePath)
        let secondRuntime = try SmeltRuntime(packagePath: packagePath)
        let mode = SmeltSelectionMode.temperature(0.8, seed: 123456789)

        XCTAssertTrue(
            firstRuntime.supportsGPUTemperatureSampling,
            "Canonical package should expose the GPU temperature sampler"
        )

        let firstToken = try firstRuntime.decodeStep(
            tokenId: 0,
            position: 0,
            selectionMode: mode
        )
        let secondToken = try secondRuntime.decodeStep(
            tokenId: 0,
            position: 0,
            selectionMode: mode
        )

        XCTAssertEqual(
            firstToken,
            secondToken,
            "GPU temperature sampling should be deterministic for a fixed seed"
        )
    }

    func testLongPromptMemoryTrimsBackToFloor() throws {
        let packagePath = try resolvePackagePath()
        let tokenizer = try SmeltTokenizer(path: "\(packagePath)/tokenizer.json")
        let (_, inputIds) = try makeBoundaryPrompt(tokenizer: tokenizer)

        let runtime = try SmeltRuntime(packagePath: packagePath)
        let baseline = runtime.memoryStats()
        let requestBatch = min(max(inputIds.count, 1), max(runtime.maxPrefillBatchSize, 1))

        try runtime.prepareForRequest(
            batchCapacity: requestBatch,
            contextCapacity: inputIds.count
        )
        let prepared = runtime.memoryStats()

        XCTAssertGreaterThanOrEqual(prepared.currentBatchCapacity, requestBatch)
        XCTAssertGreaterThanOrEqual(prepared.currentContextCapacity, inputIds.count)
        XCTAssertGreaterThan(
            prepared.totalAllocatedBytes,
            baseline.totalAllocatedBytes,
            "Preparing a long request should grow request-scoped buffers"
        )
        XCTAssertGreaterThan(
            prepared.contextScopedBytes,
            baseline.contextScopedBytes,
            "Preparing a long request should widen KV cache capacity"
        )

        runtime.resetWorkingBuffers()
        let firstToken = try runtime.prefillStep(tokenIds: inputIds, startPos: 0)
        _ = try runtime.decodeStep(tokenId: firstToken, position: Int32(inputIds.count))

        let active = runtime.memoryStats()
        runtime.trimRequestBuffers()
        let trimmed = runtime.memoryStats()

        XCTAssertLessThan(
            trimmed.totalAllocatedBytes,
            active.totalAllocatedBytes,
            "Request trim should release widened prefill/KV capacity"
        )
        XCTAssertLessThan(
            trimmed.contextScopedBytes,
            active.contextScopedBytes,
            "Request trim should shrink KV cache capacity"
        )
        XCTAssertEqual(trimmed.currentBatchCapacity, 1)
        XCTAssertEqual(trimmed.currentContextCapacity, 1)
        XCTAssertLessThanOrEqual(
            trimmed.totalAllocatedBytes,
            baseline.totalAllocatedBytes,
            "Trimmed runtime should return to its baseline footprint"
        )
    }

    func testBasePromptSnapshotMatchesFullPrompt() throws {
        let packagePath = try resolvePackagePath()
        let tokenizer = try SmeltTokenizer(path: "\(packagePath)/tokenizer.json")
        let (_, inputIds) = try makePrompt(
            tokenizer: tokenizer,
            targetMin: 160,
            targetMax: 192
        )
        let split = max(1, inputIds.count - 32)
        let baseIds = Array(inputIds[..<split])
        let suffixIds = Array(inputIds[split...])

        let reference = try decodeReferenceTokens(
            packagePath: packagePath,
            promptTokenIds: inputIds,
            maxTokens: 4
        )
        let directFullPrefill = try continuedPrefillTokens(
            packagePath: packagePath,
            capturedTokenIds: [],
            continuationTokenIds: inputIds,
            maxTokens: 4
        )
        let fullPrefill = try SmeltModel(package: packagePath).generate(tokenIds: inputIds)
        let baseGenerated = try SmeltModel(package: packagePath).generate(tokenIds: baseIds)

        let model = try SmeltModel(package: packagePath)
        let snapshot = try model.captureBasePrompt(tokenIds: baseIds)
        let continuationIds = snapshot.replayTokenIds + suffixIds
        let restoredDecode = try restoredDecodeTokens(
            packagePath: packagePath,
            snapshot: snapshot,
            suffixTokenIds: suffixIds,
            maxTokens: 4
        )
        let directContinuedPrefill = try continuedPrefillTokens(
            packagePath: packagePath,
            capturedTokenIds: Array(baseIds[..<snapshot.capturedLength]),
            continuationTokenIds: continuationIds,
            maxTokens: 4
        )
        let resumed = try model.generate(from: snapshot, tokenIds: suffixIds)
        let resumedAgain = try model.generate(from: snapshot, tokenIds: suffixIds)
        let secondModel = try SmeltModel(package: packagePath)
        let resumedFreshRuntime = try secondModel.generate(from: snapshot, tokenIds: suffixIds)
        let emptySuffix = try model.generate(from: snapshot, tokenIds: [])

        XCTAssertEqual(snapshot.promptLength, baseIds.count)
        XCTAssertGreaterThan(snapshot.byteCount, 0)
        XCTAssertLessThanOrEqual(snapshot.capturedLength, snapshot.promptLength)
        XCTAssertEqual(
            reference,
            directFullPrefill,
            "Single-step full prompt Metal prefill already diverged from decode reference"
        )
        XCTAssertEqual(
            reference,
            restoredDecode,
            "Restored snapshot diverged before Metal suffix prefill"
        )
        XCTAssertEqual(
            reference,
            directContinuedPrefill,
            "Direct startPos-prefill continuation diverged from decode reference"
        )

        let fullPrefillTokens = Array(fullPrefill.tokens.prefix(4))
        let resumedTokens = Array(resumed.tokens.prefix(4))
        let resumedAgainTokens = Array(resumedAgain.tokens.prefix(4))
        let resumedFreshTokens = Array(resumedFreshRuntime.tokens.prefix(4))
        let emptySuffixTokens = Array(emptySuffix.tokens.prefix(4))
        let baseGeneratedTokens = Array(baseGenerated.tokens.prefix(4))

        XCTAssertEqual(
            fullPrefillTokens,
            resumedTokens,
            "Restored base prompt diverged from full prompt generation"
        )
        XCTAssertEqual(
            resumedTokens,
            resumedAgainTokens,
            "Repeated restores from the same model diverged"
        )
        XCTAssertEqual(
            resumedTokens,
            resumedFreshTokens,
            "Restored base prompt diverged across fresh runtimes"
        )
        XCTAssertEqual(
            emptySuffixTokens,
            baseGeneratedTokens,
            "Empty-suffix restore did not reproduce the base prompt generation path"
        )

        let lowLevelParityHolds =
            directFullPrefill == reference
            && restoredDecode == reference
            && directContinuedPrefill == reference

        if !lowLevelParityHolds,
           let firstMismatch = try continuationTraceFirstMismatch(
                packagePath: packagePath,
                prefixTokenIds: Array(baseIds[..<snapshot.capturedLength]),
                continuationTokenIds: continuationIds
           ) {
            let fullBatchChecks = try [
                "L4.delta_conv",
            ].compactMap {
                try fullBatchMarkerDiff(
                    packagePath: packagePath,
                    prefixTokenIds: Array(baseIds[..<snapshot.capturedLength]),
                    continuationTokenIds: continuationIds,
                    label: $0
                )
            }.joined(separator: ", ")
            XCTFail("Continuation trace mismatch: \(firstMismatch) [full-batch: \(fullBatchChecks)]")
        }
    }

    func testQwen0808Layer3AttentionOProjMatchesCPUReference() throws {
        let packagePath = try resolveQwen0808PackagePath()
        let tokenizer = try SmeltTokenizer(path: "\(packagePath)/tokenizer.json")
        let inputIds = try qwenChatTokenIds(
            prompt: "The capital of France is?",
            tokenizer: tokenizer
        )
        let trace = try loadTraceMarkers(packagePath: packagePath)
        let manifest = try loadManifest(packagePath: packagePath)
        let attnCtxMarker = try XCTUnwrap(
            trace.decode.first(where: { $0.label == "L3.attn_ctx" })
        )
        let attnOutMarker = try XCTUnwrap(
            trace.decode.first(where: { $0.label == "L3.attn_out" })
        )
        let ctxCount = batchedRowStrideElements(
            for: attnCtxMarker.bufferSlot,
            manifest: manifest
        )
        let outCount = batchedRowStrideElements(
            for: attnOutMarker.bufferSlot,
            manifest: manifest
        )
        let ctxSample = try captureDecodeMarkerValues(
            packagePath: packagePath,
            tokenIds: inputIds,
            marker: attnCtxMarker,
            count: ctxCount
        )
        let outSample = try captureDecodeMarkerValues(
            packagePath: packagePath,
            tokenIds: inputIds,
            marker: attnOutMarker,
            count: outCount
        )

        XCTAssertEqual(ctxSample.token, outSample.token)

        let entry = try XCTUnwrap(
            manifest.weights.entries.first(where: { $0.name == "layers_3_self_attn_o_proj_weight" })
        )
        let weightsData = try Data(contentsOf: URL(fileURLWithPath: "\(packagePath)/weights.bin"))
        let reference = try referenceAffineOutput(
            weightsData: weightsData,
            entry: entry,
            input: ctxSample.values
        )

        var maxDiff: Float = 0
        var worstIndex = 0
        for i in 0..<min(reference.count, outSample.values.count) {
            let diff = abs(reference[i] - outSample.values[i])
            if diff > maxDiff {
                maxDiff = diff
                worstIndex = i
            }
        }

        XCTAssertLessThan(
            maxDiff,
            0.02,
            "L3 O projection diverged from CPU reference: maxDiff=\(maxDiff) at index \(worstIndex)"
        )
    }

    func testQwen0808Layer3AttentionSigmoidMulMatchesCPUReference() throws {
        let packagePath = try resolveQwen0808PackagePath()
        let tokenizer = try SmeltTokenizer(path: "\(packagePath)/tokenizer.json")
        let inputIds = try qwenChatTokenIds(
            prompt: "The capital of France is?",
            tokenizer: tokenizer
        )
        let trace = try loadTraceMarkers(packagePath: packagePath)
        let manifest = try loadManifest(packagePath: packagePath)
        let attnRawMarker = try XCTUnwrap(
            trace.decode.first(where: { $0.label == "L3.attn_raw" })
        )
        let attnCtxMarker = try XCTUnwrap(
            trace.decode.first(where: { $0.label == "L3.attn_ctx" })
        )
        let count = batchedRowStrideElements(
            for: attnRawMarker.bufferSlot,
            manifest: manifest
        )

        let rawSample = try captureDecodeMarkerValues(
            packagePath: packagePath,
            tokenIds: inputIds,
            marker: attnRawMarker,
            count: count
        )
        let gateSample = try captureDecodeSlotValues(
            packagePath: packagePath,
            tokenIds: inputIds,
            maxDispatches: attnRawMarker.dispatchCount,
            slot: 24,
            count: count
        )
        let ctxSample = try captureDecodeMarkerValues(
            packagePath: packagePath,
            tokenIds: inputIds,
            marker: attnCtxMarker,
            count: count
        )

        XCTAssertEqual(rawSample.token, gateSample.token)
        XCTAssertEqual(rawSample.token, ctxSample.token)

        var maxDiff: Float = 0
        var worstIndex = 0
        for i in 0..<count {
            let expected = rawSample.values[i] / (1 + Foundation.exp(-gateSample.values[i]))
            let diff = abs(expected - ctxSample.values[i])
            if diff > maxDiff {
                maxDiff = diff
                worstIndex = i
            }
        }

        XCTAssertLessThan(
            maxDiff,
            0.002,
            "L3 sigmoid_mul diverged from CPU reference: maxDiff=\(maxDiff) at index \(worstIndex)"
        )
    }

    func testQwen0808Prefill64MatchesDecodeTrace() throws {
        let packagePath = try resolveQwen0808PackagePath()
        let tokenizer = try SmeltTokenizer(path: "\(packagePath)/tokenizer.json")
        let (_, fullIds) = try makePrompt(
            tokenizer: tokenizer,
            targetMin: 260,
            targetMax: 400
        )
        let tokenIds = Array(fullIds.prefix(64))
        XCTAssertEqual(tokenIds.count, 64)

        let decodeRuntime = try SmeltRuntime(packagePath: packagePath)
        var decodeToken: Int32 = 0
        for (position, tokenId) in tokenIds.enumerated() {
            decodeToken = try decodeRuntime.decodeStep(tokenId: tokenId, position: Int32(position))
        }

        let prefillRuntime = try SmeltRuntime(packagePath: packagePath)
        let prefillToken = try prefillRuntime.prefillStep(tokenIds: tokenIds, startPos: 0)

        if prefillToken != decodeToken {
            let firstMismatch = try continuationTraceFirstMismatch(
                packagePath: packagePath,
                prefixTokenIds: [],
                continuationTokenIds: tokenIds
            ) ?? "no shared trace mismatch found"
            let keyChecks = try [
                "L0.delta_rec",
                "L0.delta_out",
                "L0.post_norm",
                "L0.attn_out",
                "L0.ffn_int",
                "L0.ffn_down",
                "L0.out",
                "final_norm",
            ].compactMap {
                try fullBatchMarkerDiff(
                    packagePath: packagePath,
                    prefixTokenIds: [],
                    continuationTokenIds: tokenIds,
                    label: $0
                )
            }.joined(separator: ", ")
            XCTFail(
                "0.8B prefill64 mismatch prefill=\(prefillToken) decode=\(decodeToken)"
                    + " first=\(firstMismatch) [\(keyChecks)]"
            )
        }
    }

    func testQwen4BLayer9DeltaQKVMatchesCPUReference() throws {
        let packagePath = try resolveQwen4BPackagePath()
        let trace = try loadTraceMarkers(packagePath: packagePath)
        let manifest = try loadManifest(packagePath: packagePath)

        let tokenIds: [Int32] = [
            248045, 846, 198, 760, 6511, 314, 9338, 369, 30, 271, 814, 20139,
            9495, 48575, 303, 799, 2716, 13901, 13, 271, 826, 2250, 34888,
            1429, 303, 29496, 321, 1092, 1754, 369, 16138, 1429, 364, 13,
            271, 2523, 513, 70558, 264, 9966, 1452, 3992, 1558, 383, 7905,
            35919, 13, 3855, 1330, 4222, 9640, 2195, 1062, 473, 10617, 33059,
            14134, 321, 10033, 1754, 799, 25899, 13, 248046,
        ]

        let l9InputNormMarker = try XCTUnwrap(
            trace.decode.first(where: { $0.label == "L9.input_norm" })
        )
        let l9QKVMarker = try XCTUnwrap(
            trace.decode.first(where: { $0.label == "L9.delta_qkv" })
        )

        let hiddenCount = manifest.config.hiddenSize
        let qkvCount = manifest.config.deltaQKVDim

        let layerInput = try captureDecodeMarkerValues(
            packagePath: packagePath,
            tokenIds: tokenIds,
            marker: l9InputNormMarker,
            count: hiddenCount
        )
        let qkvSample = try captureDecodeMarkerValues(
            packagePath: packagePath,
            tokenIds: tokenIds,
            marker: l9QKVMarker,
            count: qkvCount
        )

        let inputFirstNaN = layerInput.values.firstIndex(where: { !$0.isFinite })
        XCTAssertNil(
            inputFirstNaN,
            "L9.input_norm already contains non-finite values at index \(inputFirstNaN ?? -1)"
        )

        let weightsData = try Data(
            contentsOf: URL(fileURLWithPath: "\(packagePath)/weights.bin")
        )
        let entry = try XCTUnwrap(
            manifest.weights.entries.first(where: {
                $0.name == "layers_9_linear_attn_in_proj_qkv_weight"
            })
        )
        let reference = try referenceAffineOutput(
            weightsData: weightsData,
            entry: entry,
            input: layerInput.values
        )

        let qkvFirstNaN = qkvSample.values.firstIndex(where: { !$0.isFinite })
        XCTAssertNil(
            qkvFirstNaN,
            "L9.delta_qkv contains non-finite values at index \(qkvFirstNaN ?? -1)"
        )

        var maxDiff: Float = 0
        var worstIndex = 0
        for i in 0..<min(reference.count, qkvSample.values.count) {
            let diff = abs(reference[i] - qkvSample.values[i])
            if diff > maxDiff {
                maxDiff = diff
                worstIndex = i
            }
        }

        XCTAssertLessThan(
            maxDiff,
            0.05,
            "L9.delta_qkv diverged from CPU reference: maxDiff=\(maxDiff) at index \(worstIndex)"
        )
    }

    func testQwen4BLayer0DeltaQKVMatchesCPUReference() throws {
        let packagePath = try resolveQwen4BPackagePath()
        let tokenizer = try SmeltTokenizer(path: "\(packagePath)/tokenizer.json")
        let inputIds = try qwenChatTokenIds(
            prompt: "The capital of France is?",
            tokenizer: tokenizer
        )
        let manifest = try loadManifest(packagePath: packagePath)

        let hiddenCount = manifest.config.hiddenSize
        let qkvCount = manifest.config.deltaQKVDim

        let layerNormOut = try captureDecodeSlotValues(
            packagePath: packagePath,
            tokenIds: inputIds,
            maxDispatches: 2,
            slot: 8,
            count: hiddenCount
        )
        let qkvSample = try captureDecodeSlotValues(
            packagePath: packagePath,
            tokenIds: inputIds,
            maxDispatches: 3,
            slot: 2,
            count: qkvCount
        )

        XCTAssertNil(
            layerNormOut.values.firstIndex(where: { !$0.isFinite }),
            "L0 layernorm output already contains non-finite values"
        )
        XCTAssertNil(
            qkvSample.values.firstIndex(where: { !$0.isFinite }),
            "L0.delta_qkv contains non-finite values"
        )

        let weightsData = try Data(
            contentsOf: URL(fileURLWithPath: "\(packagePath)/weights.bin")
        )
        let entry = try XCTUnwrap(
            manifest.weights.entries.first(where: {
                $0.name == "layers_0_linear_attn_in_proj_qkv_weight"
            })
        )
        let reference = try referenceAffineOutput(
            weightsData: weightsData,
            entry: entry,
            input: layerNormOut.values
        )

        var maxDiff: Float = 0
        var worstIndex = 0
        for i in 0..<min(reference.count, qkvSample.values.count) {
            let diff = abs(reference[i] - qkvSample.values[i])
            if diff > maxDiff {
                maxDiff = diff
                worstIndex = i
            }
        }

        XCTAssertLessThan(
            maxDiff,
            0.05,
            "L0.delta_qkv diverged from CPU reference: maxDiff=\(maxDiff) at index \(worstIndex)"
        )
    }

    func testQwen4BLayer7AttentionOProjMatchesCPUReference() throws {
        let packagePath = try resolveQwen4BPackagePath()
        let tokenizer = try SmeltTokenizer(path: "\(packagePath)/tokenizer.json")
        let inputIds = try qwenChatTokenIds(
            prompt: "The capital of France is?",
            tokenizer: tokenizer
        )
        let trace = try loadTraceMarkers(packagePath: packagePath)
        let manifest = try loadManifest(packagePath: packagePath)
        let attnCtxMarker = try XCTUnwrap(
            trace.decode.first(where: { $0.label == "L7.attn_ctx" })
        )
        let attnOutMarker = try XCTUnwrap(
            trace.decode.first(where: { $0.label == "L7.attn_out" })
        )
        let ctxCount = batchedRowStrideElements(
            for: attnCtxMarker.bufferSlot,
            manifest: manifest
        )
        let outCount = batchedRowStrideElements(
            for: attnOutMarker.bufferSlot,
            manifest: manifest
        )
        let ctxSample = try captureDecodeMarkerValues(
            packagePath: packagePath,
            tokenIds: inputIds,
            marker: attnCtxMarker,
            count: ctxCount
        )
        let outSample = try captureDecodeMarkerValues(
            packagePath: packagePath,
            tokenIds: inputIds,
            marker: attnOutMarker,
            count: outCount
        )

        XCTAssertNil(
            ctxSample.values.firstIndex(where: { !$0.isFinite }),
            "L7.attn_ctx contains non-finite values"
        )
        XCTAssertNil(
            outSample.values.firstIndex(where: { !$0.isFinite }),
            "L7.attn_out contains non-finite values"
        )

        let entry = try XCTUnwrap(
            manifest.weights.entries.first(where: { $0.name == "layers_7_self_attn_o_proj_weight" })
        )
        let weightsData = try Data(contentsOf: URL(fileURLWithPath: "\(packagePath)/weights.bin"))
        let reference = try referenceAffineOutput(
            weightsData: weightsData,
            entry: entry,
            input: ctxSample.values
        )

        var maxDiff: Float = 0
        var worstIndex = 0
        for i in 0..<min(reference.count, outSample.values.count) {
            let diff = abs(reference[i] - outSample.values[i])
            if diff > maxDiff {
                maxDiff = diff
                worstIndex = i
            }
        }

        XCTAssertLessThan(
            maxDiff,
            0.05,
            "L7 O projection diverged from CPU reference: maxDiff=\(maxDiff) at index \(worstIndex)"
        )
    }

    func testQwen4BLayer7AttentionQueryFiniteBeforeDecode() throws {
        let packagePath = try resolveQwen4BPackagePath()
        let tokenizer = try SmeltTokenizer(path: "\(packagePath)/tokenizer.json")
        let inputIds = try qwenChatTokenIds(
            prompt: "The capital of France is?",
            tokenizer: tokenizer
        )
        let trace = try loadTraceMarkers(packagePath: packagePath)
        let attnRawMarker = try XCTUnwrap(
            trace.decode.first(where: { $0.label == "L7.attn_raw" })
        )

        let preAttention = try captureDecodeSlotValues(
            packagePath: packagePath,
            tokenIds: inputIds,
            maxDispatches: attnRawMarker.dispatchCount - 1,
            slot: attnRawMarker.bufferSlot,
            count: 4096
        )

        XCTAssertNil(
            preAttention.values.firstIndex(where: { !$0.isFinite }),
            "L7 query buffer is already non-finite before attention_decode"
        )
    }

    func testQwen4BLayer7QueryStagesStayFinite() throws {
        let packagePath = try resolveQwen4BPackagePath()
        let tokenizer = try SmeltTokenizer(path: "\(packagePath)/tokenizer.json")
        let inputIds = try qwenChatTokenIds(
            prompt: "The capital of France is?",
            tokenizer: tokenizer
        )

        let checks: [(label: String, dispatches: Int, slot: Int, count: Int)] = [
            // Layer 7 starts after L6.out at dispatch 118.
            ("L7 q_proj", 120, 20, 8192),
            ("L7 gate_split query", 123, 23, 4096),
            ("L7 q_norm", 124, 23, 4096),
            ("L7 rope_q", 126, 23, 4096),
        ]

        for check in checks {
            let sample = try captureDecodeSlotValues(
                packagePath: packagePath,
                tokenIds: inputIds,
                maxDispatches: check.dispatches,
                slot: check.slot,
                count: check.count
            )
            XCTAssertNil(
                sample.values.firstIndex(where: { !$0.isFinite }),
                "\(check.label) produced non-finite values"
            )
        }
    }

    func testQwen4BShortPromptPrefillMatchesDecode() throws {
        let packagePath = try resolveQwen4BPackagePath()
        let tokenizer = try SmeltTokenizer(path: "\(packagePath)/tokenizer.json")
        let inputIds = try qwenChatTokenIds(
            prompt: "The capital of France is?",
            tokenizer: tokenizer
        )

        let decodeToken = try decodeReferenceTokens(
            packagePath: packagePath,
            promptTokenIds: inputIds,
            maxTokens: 1
        ).first
        let prefillToken = try continuedPrefillTokens(
            packagePath: packagePath,
            capturedTokenIds: [],
            continuationTokenIds: inputIds,
            maxTokens: 1
        ).first
        XCTAssertEqual(
            decodeToken,
            prefillToken,
            "Qwen 4B short prompt prefill diverged from decode reference"
        )

        guard decodeToken != prefillToken else { return }

        let firstMismatch = try continuationTraceFirstMismatch(
            packagePath: packagePath,
            prefixTokenIds: [],
            continuationTokenIds: inputIds
        )
        XCTAssertNil(firstMismatch, firstMismatch ?? "no mismatch")
    }

    func testQwen4BLayer3QueryStagesStayFinite() throws {
        let packagePath = try resolveQwen4BPackagePath()
        let tokenizer = try SmeltTokenizer(path: "\(packagePath)/tokenizer.json")
        let inputIds = try qwenChatTokenIds(
            prompt: "The capital of France is?",
            tokenizer: tokenizer
        )

        let checks: [(label: String, dispatches: Int, slot: Int, count: Int)] = [
            // Layer 3 starts after L2.out at dispatch 49.
            ("L3 input norm", 50, 8, 2560),
            ("L3 q_proj", 51, 20, 8192),
            ("L3 gate_split query", 54, 23, 4096),
            ("L3 q_norm", 55, 23, 4096),
            ("L3 rope_q", 57, 23, 4096),
        ]

        for check in checks {
            let sample = try captureDecodeSlotValues(
                packagePath: packagePath,
                tokenIds: inputIds,
                maxDispatches: check.dispatches,
                slot: check.slot,
                count: check.count
            )
            XCTAssertNil(
                sample.values.firstIndex(where: { !$0.isFinite }),
                "\(check.label) produced non-finite values"
            )
        }
    }

}
