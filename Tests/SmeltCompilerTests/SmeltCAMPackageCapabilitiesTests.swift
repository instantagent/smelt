import Foundation
import CryptoKit
import Testing
@testable import SmeltCompiler
@testable import SmeltSchema

@Suite struct SmeltCAMPackageCapabilitiesTests {
    @Test func resolvesTextGenerationAndBenchmarkByExportFlowPortsAndGate() throws {
        let capabilities = try Self.capabilities("qwen35_text.cam")

        let run = try capabilities.resolve(.runText)
        #expect(run.exportID == "generate")
        #expect(run.flowID == "generate")
        #expect(run.inputPorts.map(Self.portShape) == ["text[encoding=utf8]"])
        #expect(run.outputPorts.map(Self.portShape) == ["text[encoding=utf8]"])
        #expect(run.selectedInputPorts.map(Self.portShape) == ["text[encoding=utf8]"])
        #expect(run.selectedOutputPorts.map(Self.portShape) == ["text[encoding=utf8]"])
        #expect(Set(run.authoredCapabilities) == ["prepare.prompt-prefix", "run.generate"])
        #expect(run.matchedGateIDs == ["startup"])
        #expect(run.matchedGateContracts.map(\.gateID) == run.matchedGateIDs)

        let bench = try capabilities.resolve(.benchDecode)
        #expect(bench.exportID == "generate")
        #expect(bench.flowID == "generate")
        #expect(bench.matchedGateIDs == ["startup"])
        let startup = try #require(bench.matchedGateContracts.first?.contract)
        #expect(startup.to?.eventType == "emit")
        #expect(startup.to?.flowID == "generate")
        #expect(startup.to?.endpoint?.endpointType == "moduleOutput")
        #expect(startup.to?.endpoint?.name == "text")
        #expect((startup.to?.predicates ?? []).map(Self.comparisonSignature) == [
            "tokens:>=:1:none",
        ])
        #expect(startup.requirements.map(Self.comparisonSignature) == [
            "elapsed:<=:115:ms",
        ])

        let serve = try capabilities.resolve(.serveText)
        #expect(serve.exportID == "generate")
        #expect(serve.flowID == "generate")

        let preparation = try capabilities.resolve(.prepareTextPromptPrefix)
        #expect(preparation.exportID == "generate")
        #expect(preparation.flowID == "generate")
        #expect(Set(preparation.authoredCapabilities) == ["prepare.prompt-prefix", "run.generate"])

        let trace = try capabilities.resolve(.traceTextGenerate)
        #expect(trace.exportID == "generate")
        #expect(trace.flowID == "generate")
    }

    @Test func resolvesExactTwoTextGenerationWithoutChangingSingleInputTextRequests() throws {
        let capabilities = try Self.capabilities("qwen35_reasoner.cam")

        let run = try capabilities.resolve(Self.twoTextRun)
        #expect(run.exportID == "review")
        #expect(run.flowID == "review")
        #expect(run.inputPorts.map(Self.namedPortShape) == [
            "candidate:text[encoding=utf8]",
            "context:text[encoding=utf8]",
        ])
        #expect(run.outputPorts.map(Self.namedPortShape) == [
            "text:text[encoding=utf8]",
        ])
        #expect(run.selectedInputPorts.map(\.portName) == ["candidate", "context"])
        #expect(run.selectedInputPorts.map(Self.portShape) == [
            "text[encoding=utf8]",
            "text[encoding=utf8]",
        ])
        #expect(run.selectedOutputPorts.map(\.portName) == ["text"])
        #expect(run.selectedOutputPorts.map(Self.portShape) == ["text[encoding=utf8]"])
        #expect(Set(run.authoredCapabilities) == ["prepare.prompt-prefix", "run.generate"])
        #expect(run.matchedGateIDs == ["startup"])

        let bench = try capabilities.resolve(Self.twoTextBench)
        #expect(bench.exportID == "review")
        #expect(bench.flowID == "review")
        let startup = try #require(bench.matchedGateContracts.first?.contract)
        #expect(startup.to?.eventType == "emit")
        #expect(startup.to?.flowID == "review")
        #expect(startup.to?.endpoint?.endpointType == "moduleOutput")
        #expect(startup.to?.endpoint?.name == "text")
        #expect((startup.to?.predicates ?? []).map(Self.comparisonSignature) == [
            "tokens:>=:1:none",
        ])
        #expect(startup.requirements.map(Self.comparisonSignature) == [
            "elapsed:<=:150:ms",
        ])

        #expect(try capabilities.resolve(Self.twoTextServe).exportID == "review")
        #expect(try capabilities.resolve(Self.twoTextTrace).flowID == "review")
        #expect(try capabilities.resolve(Self.twoTextPreparation).exportID == "review")

        for request in Self.textRequests {
            #expect(throws: SmeltCAMPackageCapabilitiesError.self) {
                _ = try capabilities.resolve(request)
            }
        }

        let unnamedTwoTextRun = SmeltCAMCapabilityRequest.exactTextToText(
            name: "unnamed two-text run",
            requiredTextInputCount: 2,
            requiredAnyExportFacts: ["run.generate"]
        )
        #expect(throws: SmeltCAMPackageCapabilitiesError.invalidRequest(
            unnamedTwoTextRun.name,
            "duplicate input shapes require explicit input port names"
        )) {
            _ = try capabilities.resolve(unnamedTwoTextRun)
        }

        let reversedInputs = try Self.capabilitiesFromDescriptorMutation("qwen35_reasoner.cam") {
            object in
            var exports = try #require(object["exports"] as? [[String: Any]])
            var inputs = try #require(exports[0]["inputs"] as? [[String: Any]])
            inputs.reverse()
            exports[0]["inputs"] = inputs
            object["exports"] = exports
        }
        let reorderedRun = try reversedInputs.resolve(Self.twoTextRun)
        #expect(reorderedRun.inputPorts.map(\.portName) == ["context", "candidate"])
        #expect(reorderedRun.selectedInputPorts.map(\.portName) == ["candidate", "context"])
    }

    @Test func exactTwoTextRequestsRejectSingleInputTextModules() throws {
        for fixture in [
            "qwen35_text.cam",
            "qwen35_fast.cam",
        ] {
            let capabilities = try Self.capabilities(fixture)
            for request in Self.twoTextRequests {
                #expect(throws: SmeltCAMPackageCapabilitiesError.self) {
                    _ = try capabilities.resolve(request)
                }
            }
        }
    }

    @Test func resolvesTextToAudioAndRejectsWrongTextRequest() throws {
        let capabilities = try Self.capabilities("qwen3_tts.cam")

        let audio = try capabilities.resolve(.runAudio)
        #expect(audio.exportID == "synth")
        #expect(audio.flowID == "synth")
        #expect(audio.inputPorts.map(Self.portShape).contains("text[encoding=utf8]"))
        #expect(audio.outputPorts.map(Self.portShape) == ["pcm[dtype=f32,rate=24khz]"])
        #expect(audio.selectedInputPorts.map(Self.portShape) == ["text[encoding=utf8]"])
        #expect(audio.selectedOutputPorts.map(Self.portShape) == ["pcm[dtype=f32,rate=24khz]"])
        #expect(Set(audio.authoredCapabilities) == [
            "prepare.voice-defaults",
            "run.stream",
            "run.synthesize",
        ])
        #expect(audio.matchedGateIDs == ["startup"])
        #expect(audio.matchedGateContracts.map(\.gateID) == audio.matchedGateIDs)
        let startup = try #require(audio.matchedGateContracts.first)
        #expect(startup.exportID == "synth")
        #expect(startup.flowID == "synth")
        #expect(startup.contract.from?.eventType == "flow.accepted")
        #expect(startup.contract.from?.flowID == "synth")
        #expect(startup.contract.to?.eventType == "emit")
        #expect(startup.contract.to?.flowID == "synth")
        #expect(startup.contract.to?.endpoint?.endpointType == "moduleOutput")
        #expect(startup.contract.to?.endpoint?.name == "audio")
        #expect((startup.contract.to?.predicates ?? []).map(Self.comparisonSignature) == [
            "duration:>=:20:ms",
            "format:==:pcm f32 24khz:none",
        ])
        #expect(startup.contract.requirements.map(Self.comparisonSignature) == [
            "elapsed:<=:400:ms",
        ])

        let serve = try capabilities.resolve(.serveAudio)
        #expect(serve.exportID == "synth")
        #expect(serve.flowID == "synth")

        let preparation = try capabilities.resolve(.prepareVoiceDefaults)
        #expect(preparation.exportID == "synth")
        #expect(preparation.flowID == "synth")
        #expect(Set(preparation.authoredCapabilities) == [
            "prepare.voice-defaults",
            "run.stream",
            "run.synthesize",
        ])

        let trace = try capabilities.resolve(.traceTextSynthesize)
        #expect(trace.exportID == "synth")
        #expect(trace.flowID == "synth")

        #expect(throws: SmeltCAMPackageCapabilitiesError.self) {
            _ = try capabilities.resolve(.runText)
        }
    }

    @Test func gateContractsAreScopedAndFilteredByRequest() throws {
        let extraAttachedGate = try Self.capabilitiesFromDescriptorMutation("qwen35_text.cam") {
            object in
            try Self.setExportGates(in: &object, exportID: "generate", gates: ["startup", "inventory"])
        }

        let run = try extraAttachedGate.resolve(.runText)
        #expect(run.matchedGateIDs == ["startup", "inventory"])
        #expect(run.matchedGateContracts.map(\.gateID) == run.matchedGateIDs)

        let bench = try extraAttachedGate.resolve(.benchDecode)
        #expect(bench.matchedGateIDs == ["startup"])
        #expect(bench.matchedGateContracts.map(\.gateID) == bench.matchedGateIDs)
        let benchStartup = try #require(bench.matchedGateContracts.first)
        #expect(benchStartup.contract.requirements.map(Self.comparisonSignature) == [
            "elapsed:<=:115:ms",
        ])

        let noAttachedElapsedGate = try Self.capabilitiesFromDescriptorMutation("qwen35_text.cam") {
            object in
            try Self.replaceGateRequirements(
                in: &object,
                gateID: "startup",
                requirements: [
                    ["subject": "quality", "relation": "<=", "value": "100", "unit": "ms"],
                ]
            )
            try Self.appendGateContract(
                in: &object,
                gate: [
                    "gateID": "unattached_elapsed",
                    "from": ["eventType": "flow.accepted", "flowID": "generate", "predicates": []],
                    "to": [
                        "eventType": "emit",
                        "flowID": "generate",
                        "endpoint": ["endpointType": "moduleOutput", "name": "text"],
                        "predicates": [
                            ["subject": "tokens", "relation": ">=", "value": "1"],
                        ],
                    ],
                    "requirements": [
                        ["subject": "elapsed", "relation": "<=", "value": "100", "unit": "ms"],
                    ],
                    "evidence": [],
                ]
            )
        }
        let stillRunnable = try noAttachedElapsedGate.resolve(.runText)
        #expect(stillRunnable.matchedGateIDs == ["startup"])
        #expect(throws: SmeltCAMPackageCapabilitiesError.noMatchingExport("bench decode")) {
            _ = try noAttachedElapsedGate.resolve(.benchDecode)
        }
    }

    @Test func dispatchTableRequestsRequireInventoryAndCompileContracts() throws {
        let fast = try Self.capabilities("qwen35_fast.cam")
        let decode = try fast.resolve(.inspectDecodeDispatchTable)
        #expect(decode.exportID == "generate")
        #expect(decode.matchedGateIDs == ["startup", "inventory"])
        let optimizerReport = try fast.resolve(.optimizerReport)
        #expect(optimizerReport.exportID == "generate")
        #expect(optimizerReport.matchedGateIDs == ["startup", "inventory"])
        let kernelLabPackage = try fast.resolve(.kernelLabPackage)
        #expect(kernelLabPackage.exportID == "generate")
        #expect(kernelLabPackage.matchedGateIDs == ["startup", "inventory"])
        let fastPrefill = try fast.resolve(.inspectPrefillDispatchTable)
        #expect(fastPrefill.exportID == "generate")
        #expect(fastPrefill.matchedGateIDs == ["startup", "inventory"])

        let text = try Self.capabilities("qwen35_text.cam")
        let prefill = try text.resolve(.inspectPrefillDispatchTable)
        #expect(prefill.exportID == "generate")
        #expect(prefill.matchedGateIDs == ["startup", "inventory"])

        let missingPrefillFile = try Self.capabilitiesFromDescriptorMutation("qwen35_text.cam") {
            object in
            try Self.replaceGateRequirements(
                in: &object,
                gateID: "inventory",
                requirements: [[
                    "subject": "package-files",
                    "relation": "include",
                    "value": "manifest.json,weights.bin,model.metallib,SmeltGenerated.swift,dispatches.bin,tokenizer.json,tokenizer.bin,module.json",
                ]]
            )
        }
        #expect(try missingPrefillFile.resolve(.inspectDecodeDispatchTable).exportID == "generate")
        #expect(throws: SmeltCAMPackageCapabilitiesError.noMatchingExport(
            "inspect prefill dispatch table"
        )) {
            _ = try missingPrefillFile.resolve(.inspectPrefillDispatchTable)
        }

        let tooSmallOptimizerInventory = try Self.capabilitiesFromDescriptorMutation("qwen35_fast.cam") {
            object in
            try Self.replaceGateRequirements(
                in: &object,
                gateID: "inventory",
                requirements: [[
                    "subject": "package-files",
                    "relation": "include",
                    "value": "dispatches.bin",
                ]]
            )
        }
        #expect(try tooSmallOptimizerInventory.resolve(.inspectDecodeDispatchTable).exportID == "generate")
        #expect(throws: SmeltCAMPackageCapabilitiesError.noMatchingExport("optimizer report")) {
            _ = try tooSmallOptimizerInventory.resolve(.optimizerReport)
        }
        #expect(throws: SmeltCAMPackageCapabilitiesError.noMatchingExport("kernel lab package")) {
            _ = try tooSmallOptimizerInventory.resolve(.kernelLabPackage)
        }

        let missingPrefillCompile = try Self.capabilitiesFromDescriptorMutation("qwen35_text.cam") {
            object in
            var compile = try #require(object["compileRequirements"] as? [[String: Any]])
            compile.removeAll { ($0["key"] as? String) == "prefill" }
            object["compileRequirements"] = compile
        }
        #expect(throws: SmeltCAMPackageCapabilitiesError.noMatchingExport(
            "inspect prefill dispatch table"
        )) {
            _ = try missingPrefillCompile.resolve(.inspectPrefillDispatchTable)
        }
    }

    @Test func benchPrefillRequiresInventoryAndCompatiblePrefillCompileContracts() throws {
        let text = try Self.capabilities("qwen35_text.cam")
        let prefill = try text.resolve(.benchPrefill)
        #expect(prefill.exportID == "generate")
        #expect(prefill.matchedGateIDs == ["startup", "inventory"])

        // qwen35_fast.cam ships the prefill runtime slice (prefill_dispatches.bin in the
        // inventory gate plus a `prefill metal batch 256` compile contract), so it resolves
        // bench prefill just like the text package. Negative coverage for the inventory and
        // plain-prefill compile preconditions is enforced by the descriptor mutations below.
        let fast = try Self.capabilities("qwen35_fast.cam")
        let fastPrefill = try fast.resolve(.benchPrefill)
        #expect(fastPrefill.exportID == "generate")
        #expect(fastPrefill.matchedGateIDs == ["startup", "inventory"])

        let verifyArgmaxPrefillCompile = try Self.capabilitiesFromDescriptorMutation(
            "qwen35_text.cam"
        ) { object in
            try Self.setCompileRequirementValue(
                in: &object,
                key: "prefill",
                value: "metal verify-argmax batch 256"
            )
        }
        let verifyArgmaxPrefill = try verifyArgmaxPrefillCompile.resolve(.benchPrefill)
        #expect(verifyArgmaxPrefill.exportID == "generate")
        #expect(verifyArgmaxPrefill.matchedGateIDs == ["startup", "inventory"])

        let allLogitsPrefillCompile = try Self.capabilitiesFromDescriptorMutation(
            "qwen35_text.cam"
        ) { object in
            try Self.setCompileRequirementValue(
                in: &object,
                key: "prefill",
                value: "metal all-logits batch 256"
            )
        }
        #expect(throws: SmeltCAMPackageCapabilitiesError.noMatchingExport("bench prefill")) {
            _ = try allLogitsPrefillCompile.resolve(.benchPrefill)
        }

        let missingPrefillFile = try Self.capabilitiesFromDescriptorMutation("qwen35_text.cam") {
            object in
            try Self.replaceGateRequirements(
                in: &object,
                gateID: "inventory",
                requirements: [[
                    "subject": "package-files",
                    "relation": "include",
                    "value": "manifest.json,weights.bin,model.metallib,SmeltGenerated.swift,dispatches.bin,tokenizer.json,tokenizer.bin,module.json",
                ]]
            )
        }
        #expect(throws: SmeltCAMPackageCapabilitiesError.noMatchingExport("bench prefill")) {
            _ = try missingPrefillFile.resolve(.benchPrefill)
        }

        let missingPrefillCompile = try Self.capabilitiesFromDescriptorMutation("qwen35_text.cam") {
            object in
            var compile = try #require(object["compileRequirements"] as? [[String: Any]])
            compile.removeAll { ($0["key"] as? String) == "prefill" }
            object["compileRequirements"] = compile
        }
        #expect(throws: SmeltCAMPackageCapabilitiesError.noMatchingExport("bench prefill")) {
            _ = try missingPrefillCompile.resolve(.benchPrefill)
        }

        let nonMetalPrefillCompile = try Self.capabilitiesFromDescriptorMutation("qwen35_text.cam") {
            object in
            try Self.setCompileRequirementValue(
                in: &object,
                key: "prefill",
                value: "coreml batch 256"
            )
        }
        #expect(throws: SmeltCAMPackageCapabilitiesError.noMatchingExport("bench prefill")) {
            _ = try nonMetalPrefillCompile.resolve(.benchPrefill)
        }

        let malformedPrefillCompile = try Self.capabilitiesFromDescriptorMutation(
            "qwen35_text.cam"
        ) { object in
            try Self.setCompileRequirementValue(
                in: &object,
                key: "prefill",
                value: "metal batch 256 extra"
            )
        }
        #expect(throws: SmeltCAMPackageCapabilitiesError.noMatchingExport("bench prefill")) {
            _ = try malformedPrefillCompile.resolve(.benchPrefill)
        }
    }

    @Test func verifyArgmaxLabRequestsRequireVerifyCompileContractAndTable() throws {
        let bonsai = try Self.capabilities("bonsai_27b_ternary.cam")
        #expect(try bonsai.resolve(.profileVerifyArgmax).exportID == "generate")
        #expect(try bonsai.resolve(.inspectVerifyArgmaxDispatchTable).exportID == "generate")

        let plainPrefillWithVerifyFile = try Self.capabilitiesFromDescriptorMutation(
            "qwen35_text.cam"
        ) { object in
            try Self.replaceGateRequirements(
                in: &object,
                gateID: "inventory",
                requirements: [[
                    "subject": "package-files",
                    "relation": "include",
                    "value": "manifest.json,weights.bin,model.metallib,SmeltGenerated.swift,dispatches.bin,prefill_dispatches.bin,prefill_verify_argmax_dispatches.bin,tokenizer.json,tokenizer.bin,module.json",
                ]]
            )
        }
        #expect(throws: SmeltCAMPackageCapabilitiesError.noMatchingExport(
            "profile verify argmax"
        )) {
            _ = try plainPrefillWithVerifyFile.resolve(.profileVerifyArgmax)
        }

        let verifyWithoutTable = try Self.capabilitiesFromDescriptorMutation(
            "bonsai_27b_ternary.cam"
        ) { object in
            try Self.replaceGateRequirements(
                in: &object,
                gateID: "inventory",
                requirements: [[
                    "subject": "package-files",
                    "relation": "include",
                    "value": "manifest.json,weights.bin,model.metallib,SmeltGenerated.swift,dispatches.bin,prefill_dispatches.bin,tokenizer.json,tokenizer.bin,module.json",
                ]]
            )
        }
        #expect(throws: SmeltCAMPackageCapabilitiesError.noMatchingExport(
            "inspect verify argmax dispatch table"
        )) {
            _ = try verifyWithoutTable.resolve(.inspectVerifyArgmaxDispatchTable)
        }
    }

    @Test func prefillParityRequiresDecodeAndPrefillInventoryBeforeRuntimeUse() throws {
        let text = try Self.capabilities("qwen35_text.cam")
        let parity = try text.resolve(.prefillParity)
        #expect(parity.exportID == "generate")
        #expect(parity.matchedGateIDs == ["startup", "inventory"])

        let missingDecodeFile = try Self.capabilitiesFromDescriptorMutation("qwen35_text.cam") {
            object in
            try Self.replaceGateRequirements(
                in: &object,
                gateID: "inventory",
                requirements: [[
                    "subject": "package-files",
                    "relation": "include",
                    "value": "manifest.json,weights.bin,model.metallib,SmeltGenerated.swift,prefill_dispatches.bin,tokenizer.json,tokenizer.bin,module.json",
                ]]
            )
        }
        #expect(try missingDecodeFile.resolve(.benchPrefill).exportID == "generate")
        #expect(throws: SmeltCAMPackageCapabilitiesError.noMatchingExport("prefill parity")) {
            _ = try missingDecodeFile.resolve(.prefillParity)
        }

        let missingPrefillFile = try Self.capabilitiesFromDescriptorMutation("qwen35_text.cam") {
            object in
            try Self.replaceGateRequirements(
                in: &object,
                gateID: "inventory",
                requirements: [[
                    "subject": "package-files",
                    "relation": "include",
                    "value": "manifest.json,weights.bin,model.metallib,SmeltGenerated.swift,dispatches.bin,tokenizer.json,tokenizer.bin,module.json",
                ]]
            )
        }
        #expect(throws: SmeltCAMPackageCapabilitiesError.noMatchingExport("prefill parity")) {
            _ = try missingPrefillFile.resolve(.prefillParity)
        }

        let missingPrefillCompile = try Self.capabilitiesFromDescriptorMutation("qwen35_text.cam") {
            object in
            var compile = try #require(object["compileRequirements"] as? [[String: Any]])
            compile.removeAll { ($0["key"] as? String) == "prefill" }
            object["compileRequirements"] = compile
        }
        #expect(throws: SmeltCAMPackageCapabilitiesError.noMatchingExport("prefill parity")) {
            _ = try missingPrefillCompile.resolve(.prefillParity)
        }
    }

    @Test func profilePrefillKernelsRequiresInventoryAndPlainPrefillCompileContracts() throws {
        let text = try Self.capabilities("qwen35_text.cam")
        let profile = try text.resolve(.profilePrefillKernels)
        #expect(profile.exportID == "generate")
        #expect(profile.matchedGateIDs == ["startup", "inventory"])

        let missingPrefillFile = try Self.capabilitiesFromDescriptorMutation("qwen35_text.cam") {
            object in
            try Self.replaceGateRequirements(
                in: &object,
                gateID: "inventory",
                requirements: [[
                    "subject": "package-files",
                    "relation": "include",
                    "value": "manifest.json,weights.bin,model.metallib,SmeltGenerated.swift,dispatches.bin,tokenizer.json,tokenizer.bin,module.json",
                ]]
            )
        }
        #expect(throws: SmeltCAMPackageCapabilitiesError.noMatchingExport(
            "profile prefill kernels"
        )) {
            _ = try missingPrefillFile.resolve(.profilePrefillKernels)
        }

        let allLogitsPrefillCompile = try Self.capabilitiesFromDescriptorMutation(
            "qwen35_text.cam"
        ) { object in
            try Self.setCompileRequirementValue(
                in: &object,
                key: "prefill",
                value: "metal all-logits batch 256"
            )
        }
        #expect(throws: SmeltCAMPackageCapabilitiesError.noMatchingExport(
            "profile prefill kernels"
        )) {
            _ = try allLogitsPrefillCompile.resolve(.profilePrefillKernels)
        }

        let missingPrefillCompile = try Self.capabilitiesFromDescriptorMutation("qwen35_text.cam") {
            object in
            var compile = try #require(object["compileRequirements"] as? [[String: Any]])
            compile.removeAll { ($0["key"] as? String) == "prefill" }
            object["compileRequirements"] = compile
        }
        #expect(throws: SmeltCAMPackageCapabilitiesError.noMatchingExport(
            "profile prefill kernels"
        )) {
            _ = try missingPrefillCompile.resolve(.profilePrefillKernels)
        }
    }

    @Test func profileDecodeKernelsRequiresDecodeInventory() throws {
        let text = try Self.capabilities("qwen35_text.cam")
        let profile = try text.resolve(.profileDecodeKernels)
        #expect(profile.exportID == "generate")
        #expect(profile.matchedGateIDs == ["startup", "inventory"])

        let missingDecodeFile = try Self.capabilitiesFromDescriptorMutation("qwen35_text.cam") {
            object in
            try Self.replaceGateRequirements(
                in: &object,
                gateID: "inventory",
                requirements: [[
                    "subject": "package-files",
                    "relation": "include",
                    "value": "manifest.json,weights.bin,model.metallib,SmeltGenerated.swift,prefill_dispatches.bin,tokenizer.json,tokenizer.bin,module.json",
                ]]
            )
        }
        #expect(throws: SmeltCAMPackageCapabilitiesError.noMatchingExport(
            "profile decode kernels"
        )) {
            _ = try missingDecodeFile.resolve(.profileDecodeKernels)
        }

        let missingPrefillFile = try Self.capabilitiesFromDescriptorMutation("qwen35_text.cam") {
            object in
            try Self.replaceGateRequirements(
                in: &object,
                gateID: "inventory",
                requirements: [[
                    "subject": "package-files",
                    "relation": "include",
                    "value": "manifest.json,weights.bin,model.metallib,SmeltGenerated.swift,dispatches.bin,tokenizer.json,tokenizer.bin,module.json",
                ]]
            )
        }
        #expect(try missingPrefillFile.resolve(.profileDecodeKernels).exportID == "generate")
    }

    @Test func benchDecodeRequiresObservedFirstTextOutputGate() throws {
        let wrongOutput = try Self.capabilitiesFromDescriptorMutation("qwen35_text.cam") { object in
            try Self.appendExportOutput(
                in: &object,
                exportID: "generate",
                portName: "debug_text",
                typeName: "text",
                attributes: ["encoding": "utf8"]
            )
            try Self.replaceGateToEndpoint(
                in: &object,
                gateID: "startup",
                endpoint: ["endpointType": "moduleOutput", "name": "debug_text"]
            )
        }
        #expect(try wrongOutput.resolve(.runText).exportID == "generate")
        #expect(throws: SmeltCAMPackageCapabilitiesError.noMatchingExport("bench decode")) {
            _ = try wrongOutput.resolve(.benchDecode)
        }

        let wrongEndpointType = try Self.capabilitiesFromDescriptorMutation("qwen35_text.cam") {
            object in
            try Self.replaceGateToEndpoint(
                in: &object,
                gateID: "startup",
                endpoint: [
                    "endpointType": "nodePort",
                    "nodeID": "detokenizer",
                    "portName": "text",
                ]
            )
        }
        #expect(try wrongEndpointType.resolve(.runText).exportID == "generate")
        #expect(throws: SmeltCAMPackageCapabilitiesError.noMatchingExport("bench decode")) {
            _ = try wrongEndpointType.resolve(.benchDecode)
        }

        let missingPredicate = try Self.capabilitiesFromDescriptorMutation("qwen35_text.cam") {
            object in
            try Self.replaceGateToPredicates(in: &object, gateID: "startup", predicates: [])
        }
        #expect(try missingPredicate.resolve(.runText).exportID == "generate")
        #expect(throws: SmeltCAMPackageCapabilitiesError.noMatchingExport("bench decode")) {
            _ = try missingPredicate.resolve(.benchDecode)
        }

        let wrongPredicate = try Self.capabilitiesFromDescriptorMutation("qwen35_text.cam") {
            object in
            try Self.replaceGateToPredicates(
                in: &object,
                gateID: "startup",
                predicates: [["subject": "tokens", "relation": "==", "value": "0"]]
            )
        }
        #expect(try wrongPredicate.resolve(.runText).exportID == "generate")
        #expect(throws: SmeltCAMPackageCapabilitiesError.noMatchingExport("bench decode")) {
            _ = try wrongPredicate.resolve(.benchDecode)
        }

        let wrongEventType = try Self.capabilitiesFromDescriptorMutation("qwen35_text.cam") {
            object in
            try Self.replaceGateTo(
                in: &object,
                gateID: "startup",
                to: [
                    "eventType": "input",
                    "flowID": "generate",
                    "endpoint": ["endpointType": "moduleInput", "name": "prompt"],
                    "predicates": [["subject": "tokens", "relation": ">=", "value": "1"]],
                ]
            )
        }
        #expect(try wrongEventType.resolve(.runText).exportID == "generate")
        #expect(throws: SmeltCAMPackageCapabilitiesError.noMatchingExport("bench decode")) {
            _ = try wrongEventType.resolve(.benchDecode)
        }
    }

    @Test func runtimeAudioRequestsRequireObservedFirstPublicAudioGate() throws {
        let wrongPublicOutput = try Self.capabilitiesFromDescriptorMutation("qwen3_tts.cam") {
            object in
            try Self.appendExportOutput(
                in: &object,
                exportID: "synth",
                portName: "debug_text",
                typeName: "text",
                attributes: ["encoding": "utf8"]
            )
            try Self.replaceGateToEndpoint(
                in: &object,
                gateID: "startup",
                endpoint: ["endpointType": "moduleOutput", "name": "debug_text"]
            )
        }
        try Self.expectRuntimeAudioRejectsButPreparationResolves(wrongPublicOutput)

        let wrongEndpointType = try Self.capabilitiesFromDescriptorMutation("qwen3_tts.cam") {
            object in
            try Self.replaceGateToEndpoint(
                in: &object,
                gateID: "startup",
                endpoint: [
                    "endpointType": "nodePort",
                    "nodeID": "codec-decoder",
                    "portName": "audio",
                ]
            )
        }
        try Self.expectRuntimeAudioRejectsButPreparationResolves(wrongEndpointType)

        let wrongFromEvent = try Self.capabilitiesFromDescriptorMutation("qwen3_tts.cam") {
            object in
            try Self.replaceGateFrom(
                in: &object,
                gateID: "startup",
                from: [
                    "eventType": "input",
                    "flowID": "synth",
                    "endpoint": ["endpointType": "moduleInput", "name": "text"],
                    "predicates": [],
                ]
            )
        }
        try Self.expectRuntimeAudioRejectsButPreparationResolves(wrongFromEvent)

        let wrongToEvent = try Self.capabilitiesFromDescriptorMutation("qwen3_tts.cam") {
            object in
            try Self.replaceGateTo(
                in: &object,
                gateID: "startup",
                to: [
                    "eventType": "input",
                    "flowID": "synth",
                    "endpoint": ["endpointType": "moduleInput", "name": "text"],
                    "predicates": [
                        ["subject": "duration", "relation": ">=", "value": "20", "unit": "ms"],
                        ["subject": "format", "relation": "==", "value": "pcm f32 24khz"],
                    ],
                ]
            )
        }
        try Self.expectRuntimeAudioRejectsButPreparationResolves(wrongToEvent)

        let missingElapsedRequirement = try Self.capabilitiesFromDescriptorMutation(
            "qwen3_tts.cam"
        ) { object in
            try Self.replaceGateRequirements(
                in: &object,
                gateID: "startup",
                requirements: [
                    ["subject": "quality", "relation": ">=", "value": "1"],
                ]
            )
        }
        try Self.expectRuntimeAudioRejectsButPreparationResolves(missingElapsedRequirement)

        let missingDurationPredicate = try Self.capabilitiesFromDescriptorMutation(
            "qwen3_tts.cam"
        ) { object in
            try Self.replaceGateToPredicates(
                in: &object,
                gateID: "startup",
                predicates: [
                    ["subject": "format", "relation": "==", "value": "pcm f32 24khz"],
                ]
            )
        }
        try Self.expectRuntimeAudioRejectsButPreparationResolves(missingDurationPredicate)

        let wrongFormatPredicate = try Self.capabilitiesFromDescriptorMutation("qwen3_tts.cam") {
            object in
            try Self.replaceGateToPredicates(
                in: &object,
                gateID: "startup",
                predicates: [
                    ["subject": "duration", "relation": ">=", "value": "20", "unit": "ms"],
                    ["subject": "format", "relation": "==", "value": "pcm f32 16khz"],
                ]
            )
        }
        try Self.expectRuntimeAudioRejectsButPreparationResolves(wrongFormatPredicate)
    }

    @Test func runtimeAudioGateThresholdsAreDefinedByCAM() throws {
        let changedThresholds = try Self.capabilitiesFromDescriptorMutation("qwen3_tts.cam") {
            object in
            try Self.replaceGateRequirements(
                in: &object,
                gateID: "startup",
                requirements: [
                    ["subject": "elapsed", "relation": "<=", "value": "250", "unit": "ms"],
                ]
            )
            try Self.replaceGateToPredicates(
                in: &object,
                gateID: "startup",
                predicates: [
                    ["subject": "duration", "relation": ">=", "value": "5", "unit": "ms"],
                    ["subject": "format", "relation": "==", "value": "pcm f32 24khz"],
                ]
            )
        }

        for request in Self.runtimeAudioRequests {
            #expect(try changedThresholds.resolve(request).exportID == "synth")
        }
    }

    @Test func traceRouteWitnessV6NamesRouteOwnedFields() throws {
        let textCapabilities = try Self.capabilities("qwen35_text.cam")
        let text = try textCapabilities.resolve(.traceTextGenerate)
        #expect(
            text.traceRouteWitnessV6(
                camSemanticSHA256: textCapabilities.camSemanticSHA256,
                exportABISHA256: textCapabilities.exportABISHA256
            )
                == "cam-route:v6;"
                + "cam=\(textCapabilities.camSemanticSHA256);"
                + "exportABI=\(textCapabilities.exportABISHA256);"
                + "export=generate;"
                + "flow=generate;"
                + "gates=startup;"
                + "capabilities=prepare.prompt-prefix%2Crun.generate;"
                + "inputs=prompt%3Atext%7Bencoding%3Dutf8%7D%3Arequired;"
                + "outputs=text%3Atext%7Bencoding%3Dutf8%7D%3Arequired"
        )

        let ttsCapabilities = try Self.capabilities("qwen3_tts.cam")
        let tts = try ttsCapabilities.resolve(.traceTextSynthesize)
        #expect(
            tts.traceRouteWitnessV6(
                camSemanticSHA256: ttsCapabilities.camSemanticSHA256,
                exportABISHA256: ttsCapabilities.exportABISHA256
            )
                == "cam-route:v6;"
                + "cam=\(ttsCapabilities.camSemanticSHA256);"
                + "exportABI=\(ttsCapabilities.exportABISHA256);"
                + "export=synth;"
                + "flow=synth;"
                + "gates=startup;"
                + "capabilities=prepare.voice-defaults%2Crun.stream%2Crun.synthesize;"
                + "inputs=speaker%3Avoice-id%7B%7D%3Aoptional%2Ctext%3Atext%7Bencoding%3Dutf8%7D%3Arequired;"
                + "outputs=audio%3Apcm%7Bdtype%3Df32%2Crate%3D24khz%7D%3Arequired"
        )
    }

    @Test func traceRouteWitnessV6UsesRouteShapeNotRequestAliases() throws {
        let textCapabilities = try Self.capabilities("qwen35_text.cam")
        let textWitnesses = try [
            textCapabilities.resolve(.runText),
            textCapabilities.resolve(.serveText),
            textCapabilities.resolve(.traceTextGenerate),
        ].map {
            $0.traceRouteWitnessV6(
                camSemanticSHA256: textCapabilities.camSemanticSHA256,
                exportABISHA256: textCapabilities.exportABISHA256
            )
        }
        #expect(Set(textWitnesses).count == 1)

        let audioCapabilities = try Self.capabilities("qwen3_tts.cam")
        let audioWitnesses = try [
            audioCapabilities.resolve(.runAudio),
            audioCapabilities.resolve(.serveAudio),
            audioCapabilities.resolve(.traceTextSynthesize),
        ].map {
            $0.traceRouteWitnessV6(
                camSemanticSHA256: audioCapabilities.camSemanticSHA256,
                exportABISHA256: audioCapabilities.exportABISHA256
            )
        }
        #expect(Set(audioWitnesses).count == 1)

        let original = try textCapabilities.resolve(.runText)
        let alternate = SmeltCAMPackageCapabilities.Decision(
            exportID: "generate_alt",
            flowID: "generate_alt",
            inputPorts: original.inputPorts,
            outputPorts: original.outputPorts,
            selectedInputPorts: original.selectedInputPorts,
            selectedOutputPorts: original.selectedOutputPorts,
            matchedGateIDs: original.matchedGateIDs,
            matchedGateContracts: original.matchedGateContracts,
            authoredCapabilities: original.authoredCapabilities
        )
        let originalWitness = original.traceRouteWitnessV6(
            camSemanticSHA256: textCapabilities.camSemanticSHA256,
            exportABISHA256: textCapabilities.exportABISHA256
        )
        let alternateWitness = alternate.traceRouteWitnessV6(
            camSemanticSHA256: textCapabilities.camSemanticSHA256,
            exportABISHA256: textCapabilities.exportABISHA256
        )
        #expect(originalWitness != alternateWitness)
        #expect(originalWitness.contains("export=generate;flow=generate;"))
        #expect(alternateWitness.contains("export=generate_alt;flow=generate_alt;"))
    }

    @Test func traceRouteWitnessV6UsesOnlyRouteIdentityFields() throws {
        let cases: [(String, SmeltCAMCapabilityRequest)] = [
            ("qwen35_text.cam", .traceTextGenerate),
            ("qwen3_tts.cam", .traceTextSynthesize),
        ]

        for (fixture, request) in cases {
            let capabilities = try Self.capabilities(fixture)
            let decision = try capabilities.resolve(request)
            let witness = decision.traceRouteWitnessV6(
                camSemanticSHA256: capabilities.camSemanticSHA256,
                exportABISHA256: capabilities.exportABISHA256
            )
            let normalized = witness.lowercased()

            #expect(witness.hasPrefix("cam-route:v6;"))
            #expect(!witness.contains("manifestBridgeRoute"))
            #expect(!witness.contains("request="))
            #expect(!witness.contains("requiredInputs="))
            #expect(!witness.contains("requiredOutputs="))
            #expect(!witness.contains(" "))
            #expect(!witness.contains("temporaryRuntimeWitness"))
            #expect(!witness.contains("temporary-"))
            for banned in [
                "trace text generation",
                "trace%20text%20generation",
                "trace text synthesis",
                "trace%20text%20synthesis",
                "llm",
                "tts",
                "asr",
                "qwen",
                "whisper",
                "kind",
                "policymode",
                "architecture",
            ] {
                #expect(!normalized.contains(banned))
            }
        }

        let capabilities = try Self.capabilities("qwen35_text.cam")
        let original = try capabilities.resolve(.traceTextGenerate)
        let escaped = SmeltCAMPackageCapabilities.Decision(
            exportID: "generate;text=bad",
            flowID: original.flowID,
            inputPorts: original.inputPorts,
            outputPorts: original.outputPorts,
            selectedInputPorts: original.selectedInputPorts,
            selectedOutputPorts: original.selectedOutputPorts,
            matchedGateIDs: original.matchedGateIDs,
            matchedGateContracts: original.matchedGateContracts,
            authoredCapabilities: original.authoredCapabilities
        )
        let witness = escaped.traceRouteWitnessV6(
            camSemanticSHA256: capabilities.camSemanticSHA256,
            exportABISHA256: capabilities.exportABISHA256
        )
        #expect(witness.contains("export=generate%3Btext%3Dbad"))
        #expect(!witness.contains(";text=bad"))
    }

    @Test func runtimeContractBodyIsStableAcrossRequestAliasesForSameRoute() throws {
        let textCapabilities = try Self.capabilities("qwen35_text.cam")
        let textBodies = try [
            textCapabilities.resolve(.runText),
            textCapabilities.resolve(.serveText),
            textCapabilities.resolve(.traceTextGenerate),
        ].map {
            try textCapabilities.runtimeContractSignature(for: $0).body
        }
        #expect(Set(textBodies).count == 1)

        let audioCapabilities = try Self.capabilities("qwen3_tts.cam")
        let audioBodies = try [
            audioCapabilities.resolve(.runAudio),
            audioCapabilities.resolve(.serveAudio),
            audioCapabilities.resolve(.traceTextSynthesize),
        ].map {
            try audioCapabilities.runtimeContractSignature(for: $0).body
        }
        #expect(Set(audioBodies).count == 1)
    }

    @Test func runtimeContractCapturesFlowGateRateFeedbackAndQuantFacts() throws {
        let textCapabilities = try Self.capabilities("qwen35_text.cam")
        let textContract = try textCapabilities.runtimeContractSignature(
            for: textCapabilities.resolve(.runText)
        )
        #expect(textContract.lines.first == "cam-runtime-contract:v1")
        #expect(textContract.provenance.contains("descriptor-schema:smelt.module.package_descriptor.v2"))
        #expect(textContract.provenance.contains("descriptor-version:2"))
        #expect(textContract.provenance.contains("cam:\(textCapabilities.camSemanticSHA256)"))
        #expect(textContract.provenance.contains("export-abi:\(textCapabilities.exportABISHA256)"))
        #expect(textContract.body.contains("capability:run.generate"))
        #expect(textContract.body.contains("selected-output:text:text{encoding=utf8}:required"))
        #expect(textContract.body.contains("stop:max-steps:512"))
        #expect(textContract.body.contains("gate:require:elapsed:<=:115:ms"))
        #expect(textContract.body.contains("feedback:node:n0:tokens->node:n2:tokens"))
        #expect(textContract.body.contains("quant:default:storage=affine-u4:group=64:compute=none:priority=none:calibration=none:resolution=declared-tensor"))

        let audioCapabilities = try Self.capabilities("qwen3_tts.cam")
        let audioContract = try audioCapabilities.runtimeContractSignature(
            for: audioCapabilities.resolve(.runAudio)
        )
        #expect(audioContract.body.contains("selected-output:audio:pcm{dtype=f32,rate=24khz}:required"))
        #expect(audioContract.body.contains("gate:to:emit:flow=selected:export=none:endpoint=moduleOutput:audio:signal=none:predicates(duration:>=:20:ms,format:==:pcm f32 24khz:none)"))
        #expect(audioContract.body.contains("feedback:node:n2:codec_token->node:n3:codec_token"))
    }

    @Test func runtimeContractBodyChangesForRouteConstructionDrift() throws {
        let original = try Self.capabilities("qwen3_tts.cam")
        let originalBody = try original.runtimeContractSignature(
            for: original.resolve(.runAudio)
        ).body

        let changedThresholds = try Self.capabilitiesFromDescriptorMutation("qwen3_tts.cam") {
            object in
            try Self.replaceGateRequirements(
                in: &object,
                gateID: "startup",
                requirements: [
                    ["subject": "elapsed", "relation": "<=", "value": "250", "unit": "ms"],
                ]
            )
            try Self.replaceGateToPredicates(
                in: &object,
                gateID: "startup",
                predicates: [
                    ["subject": "duration", "relation": ">=", "value": "5", "unit": "ms"],
                    ["subject": "format", "relation": "==", "value": "pcm f32 24khz"],
                ]
            )
        }
        let changedThresholdBody = try changedThresholds.runtimeContractSignature(
            for: changedThresholds.resolve(.runAudio)
        ).body
        #expect(changedThresholdBody != originalBody)
        #expect(changedThresholdBody.contains("gate:require:elapsed:<=:250:ms"))
        #expect(changedThresholdBody.contains("gate:to:emit:flow=selected:export=none:endpoint=moduleOutput:audio:signal=none:predicates(duration:>=:5:ms,format:==:pcm f32 24khz:none)"))

        let changedRate = try Self.capabilitiesFromDescriptorMutation("qwen3_tts.cam") {
            object in
            var exports = try #require(object["exports"] as? [[String: Any]])
            try Self.mutatePortAttributes(
                in: &exports[0],
                portListKey: "outputs",
                index: 0,
                attributes: ["dtype": "f32", "rate": "16khz"]
            )
            object["exports"] = exports
            try Self.mutateGraphNodePortAttributes(
                in: &object,
                nodeID: "codec-decoder",
                portListKey: "outputs",
                portName: "audio",
                attributes: ["dtype": "f32", "rate": "16khz"]
            )
            try Self.mutateGraphEdgeValueType(
                in: &object,
                typeName: "pcm",
                attributes: ["dtype": "f32", "rate": "16khz"]
            ) { from, to in
                from["endpointType"] as? String == "nodePort"
                    && from["nodeID"] as? String == "codec-decoder"
                    && from["portName"] as? String == "audio"
                    && to["endpointType"] as? String == "moduleOutput"
                    && to["name"] as? String == "audio"
            }
            try Self.replaceGateToPredicates(
                in: &object,
                gateID: "startup",
                predicates: [
                    ["subject": "duration", "relation": ">=", "value": "20", "unit": "ms"],
                    ["subject": "format", "relation": "==", "value": "pcm f32 16khz"],
                ]
            )
        }
        let changedRateBody = try changedRate.runtimeContractSignature(
            for: changedRate.resolve(.runAudio)
        ).body
        #expect(changedRateBody != originalBody)
        #expect(changedRateBody.contains("selected-output:audio:pcm{dtype=f32,rate=16khz}:required"))
    }

    @Test func runtimeContractBodyIgnoresUnmatchedGates() throws {
        let original = try Self.capabilities("qwen35_text.cam")
        let originalContract = try original.runtimeContractSignature(
            for: original.resolve(.runText)
        )

        let extraGate = try Self.capabilitiesFromDescriptorMutation("qwen35_text.cam") { object in
            try Self.appendGateContract(
                in: &object,
                gate: [
                    "gateID": "unattached_elapsed",
                    "from": ["eventType": "flow.accepted", "flowID": "generate", "predicates": []],
                    "to": [
                        "eventType": "emit",
                        "flowID": "generate",
                        "endpoint": ["endpointType": "moduleOutput", "name": "text"],
                        "predicates": [
                            ["subject": "tokens", "relation": ">=", "value": "1"],
                        ],
                    ],
                    "requirements": [
                        ["subject": "elapsed", "relation": "<=", "value": "999", "unit": "ms"],
                    ],
                    "evidence": [],
                ]
            )
        }
        let extraGateContract = try extraGate.runtimeContractSignature(
            for: extraGate.resolve(.runText)
        )
        #expect(extraGateContract.body == originalContract.body)
        #expect(extraGateContract.provenance == originalContract.provenance)
    }

    @Test func runtimeContractBodyAvoidsSelectorAndIdentityFields() throws {
        let cases: [(String, SmeltCAMCapabilityRequest)] = [
            ("qwen35_text.cam", .runText),
            ("qwen3_tts.cam", .runAudio),
        ]
        for (fixture, request) in cases {
            let capabilities = try Self.capabilities(fixture)
            let body = try capabilities.runtimeContractSignature(
                for: capabilities.resolve(request)
            ).body.joined(separator: "\n").lowercased()
            for banned in [
                "qwen",
                "stable_audio",
                "stable-audio",
                "whisper",
                "llm",
                "tts",
                "asr",
                "policymode",
                "architecture",
                "runtimedescriptor",
                "moduleid",
                "package_projection",
                "packageprojection",
                "module.json",
                "manifest.json",
                "tokenizer.json",
                "weights.bin",
            ] {
                #expect(!body.contains(banned), "\(fixture): \(banned)")
            }
            #expect(!body.contains(capabilities.camSemanticSHA256))
            #expect(!body.contains(capabilities.exportABISHA256))
        }
    }

    @Test func runtimeContractBodyHashesStayPinnedAcrossRouteFactRefactor() throws {
        let cases: [(String, SmeltCAMCapabilityRequest, String)] = [
            (
                "qwen35_text.cam",
                .runText,
                "61cf7390a5610d27053de9ada937b9a087bd5ee7448a468d646220d70ec78ccb"
            ),
            (
                "qwen3_tts.cam",
                .runAudio,
                "1c404f02cec0755956ab465f1f8fce87014707729359302967ea5769b6da0fcc"
            ),
            (
                "ds4_heavy_quant.cam",
                .runText,
                "dfc5172de2cae76a6aa028394b74d2a9a7536c2494d364549dbb8bdaad413454"
            ),
        ]

        for (fixture, request, expected) in cases {
            let capabilities = try Self.capabilities(fixture)
            let contract = try capabilities.runtimeContractSignature(
                for: capabilities.resolve(request)
            )
            #expect(
                Self.sha256Hex(contract.body.joined(separator: "\n")) == expected,
                "\(fixture)"
            )
        }
    }

    @Test func runtimeAssemblyFeatureContractDerivesGraphFeaturesForCompletionFixtures() throws {
        let text = try Self.runtimeAssemblyFeatureContract("qwen35_text.cam", .runText)
        Self.expectGraphAndSurfaceFeaturesAreSplit(text)
        #expect(text.configuredGraphFeatureSet.contains("flow.phase.step"))
        #expect(text.configuredGraphFeatureSet.contains("flow.call.node"))
        #expect(text.configuredGraphFeatureSet.contains("graph.impl.compiled"))
        #expect(text.configuredGraphFeatureSet.contains("graph.feedback"))
        #expect(text.configuredGraphFeatureSet.contains("graph.edge.tokens"))
        #expect(text.configuredGraphFeatureSet.contains("block.transformer"))
        #expect(text.configuredGraphFeatureSet.contains("block.transformer.delta"))
        #expect(text.configuredGraphFeatureSet.contains("block.transformer.attention"))
        #expect(text.configuredGraphFeatureSet.contains("block.transformer.rope.neox"))
        #expect(text.configuredGraphFeatureSet.contains("block.transformer.ffn.swiglu"))
        #expect(text.configuredGraphFeatureSet.contains("block.transformer.vocab.tied-head"))
        #expect(text.configuredGraphFeatureSet.contains("quant.storage.affine-u4"))
        #expect(text.configuredGraphFeatureSet.contains("compile.backend.metal"))
        #expect(text.configuredGraphFeatureSet.contains("compile.prefill"))
        #expect(text.featureSet.contains("io.text"))
        #expect(text.featureSet.contains("gate.startup"))
        #expect(text.featureSet.contains("gate.subject.elapsed"))
        #expect(text.featureSet.contains("gate.subject.tokens"))

        let speech = try Self.runtimeAssemblyFeatureContract("qwen3_tts.cam", .runAudio)
        Self.expectGraphAndSurfaceFeaturesAreSplit(speech)
        #expect(speech.configuredGraphFeatureSet.contains("block.frontend"))
        #expect(speech.configuredGraphFeatureSet.contains("block.frontend.speaker-conditioning"))
        #expect(speech.configuredGraphFeatureSet.contains("block.transformer"))
        #expect(speech.configuredGraphFeatureSet.contains("block.codec-decoder"))
        #expect(speech.configuredGraphFeatureSet.contains("block.codec-decoder.streaming"))
        #expect(speech.configuredGraphFeatureSet.contains("block.requirement.codec-feedback"))
        #expect(speech.configuredGraphFeatureSet.contains("block.requirement.codebooks"))
        #expect(speech.configuredGraphFeatureSet.contains("graph.codebooks"))
        #expect(speech.configuredGraphFeatureSet.contains("graph.feedback"))
        #expect(speech.configuredGraphFeatureSet.contains("artifact.role.sidecar"))
        #expect(speech.configuredGraphFeatureSet.contains("artifact.role.compiled-inline"))
        #expect(speech.featureSet.contains("io.pcm"))
        #expect(speech.featureSet.contains("gate.subject.duration"))
        #expect(speech.featureSet.contains("gate.subject.format"))
    }

    @Test func runtimeAssemblyFeatureContractCoversDS4WithoutChangingFeatureAdmission() throws {
        let capabilities = try Self.capabilities("ds4_heavy_quant.cam")
        let contract = try capabilities.runtimeAssemblyFeatureContract(
            for: capabilities.resolve(.runText)
        )
        Self.expectGraphAndSurfaceFeaturesAreSplit(contract)
        for feature in [
            "block.transformer.rope.yarn",
            "block.transformer.moe",
            "block.transformer.router",
            "block.transformer.expert",
            "quant.storage.gptq",
            "quant.calibration.gptq",
            "quant.calibration.rank-gate",
            "quant.calibration.perplexity-gate",
            "compile.generated-kernels",
            "compile.memory-bound",
        ] {
            #expect(contract.configuredGraphFeatureSet.contains(feature), "\(feature)")
        }
        #expect(capabilities.featureAdmission.unsupportedFeatureSet == [
            "compile.generated-kernels",
            "compile.memory-bound",
            "gate.quant-quality",
            "quant.calibration.gptq",
            "quant.storage.gptq",
            "transformer.moe.expert",
            "transformer.moe.router",
            "transformer.rope.yarn",
        ])
    }

    @Test func runtimeAssemblyFeatureContractIsRouteStableAndMutationSensitive() throws {
        let capabilities = try Self.capabilities("qwen35_text.cam")
        let baseline = try capabilities.runtimeAssemblyFeatureContract(for: capabilities.resolve(.runText))
        let aliases = try [
            capabilities.resolve(.serveText),
            capabilities.resolve(.traceTextGenerate),
            capabilities.resolve(.prepareTextPromptPrefix),
        ].map { try capabilities.runtimeAssemblyFeatureContract(for: $0) }
        for alias in aliases {
            #expect(alias.configuredGraphFeatureSet == baseline.configuredGraphFeatureSet)
            #expect(alias.featureSet == baseline.featureSet)
        }

        let unusedCodecBlock = try Self.capabilitiesFromDescriptorMutation("qwen35_text.cam") {
            object in
            var blocks = try #require(object["blocks"] as? [[String: Any]])
            let audioBlocks = try #require(
                Self.descriptorJSONObject("qwen3_tts.cam")["blocks"] as? [[String: Any]]
            )
            var codec = try #require(audioBlocks.first {
                ($0["operatorName"] as? String) == "codec-decoder"
            })
            codec["blockID"] = "unused_codec_probe"
            blocks.append(codec)
            object["blocks"] = blocks
        }
        let unusedCodec = try unusedCodecBlock.runtimeAssemblyFeatureContract(
            for: unusedCodecBlock.resolve(.runText)
        )
        #expect(unusedCodec.configuredGraphFeatureSet == baseline.configuredGraphFeatureSet)

        let noFeedback = try Self.capabilitiesFromDescriptorMutation("qwen35_text.cam") { object in
            object["feedbackEdges"] = []
        }
        let noFeedbackContract = try noFeedback.runtimeAssemblyFeatureContract(
            for: noFeedback.resolve(.runText)
        )
        #expect(!noFeedbackContract.configuredGraphFeatureSet.contains("graph.feedback"))
        #expect(noFeedbackContract.configuredGraphFeatureSet != baseline.configuredGraphFeatureSet)

        let codecNotStreaming = try Self.capabilitiesFromDescriptorMutation("qwen3_tts.cam") {
            object in
            try Self.mutateBlockCodecStreaming(
                in: &object,
                blockID: "codec-decoder",
                streaming: false
            )
        }
        let codecContract = try codecNotStreaming.runtimeAssemblyFeatureContract(
            for: codecNotStreaming.resolve(.runAudio)
        )
        #expect(!codecContract.configuredGraphFeatureSet.contains("block.codec-decoder.streaming"))

        let noGeneratedKernels = try Self.capabilitiesFromDescriptorMutation("ds4_heavy_quant.cam") {
            object in
            var compile = try #require(object["compileRequirements"] as? [[String: Any]])
            compile.removeAll { ($0["key"] as? String) == "generated-kernels" }
            object["compileRequirements"] = compile
        }
        let noGeneratedContract = try noGeneratedKernels.runtimeAssemblyFeatureContract(
            for: noGeneratedKernels.resolve(.runText)
        )
        #expect(!noGeneratedContract.configuredGraphFeatureSet.contains("compile.generated-kernels"))
    }

    @Test func runtimeAssemblyFeatureCodesRejectSelectorVocabulary() throws {
        let cases: [(String, SmeltCAMCapabilityRequest)] = [
            ("qwen35_text.cam", .runText),
            ("qwen3_tts.cam", .runAudio),
            ("ds4_heavy_quant.cam", .runText),
        ]
        for (fixture, request) in cases {
            let contract = try Self.runtimeAssemblyFeatureContract(fixture, request)
            for feature in contract.configuredGraphFeatureSet + contract.featureSet {
                #expect(Self.selectorFeatureViolations(feature).isEmpty, "\(fixture): \(feature)")
                #expect(!feature.contains("target"), "\(fixture): \(feature)")
            }
            #expect(contract.configuredGraphFeatureSet.contains { !$0.hasPrefix("io.") && !$0.hasPrefix("gate.") })
        }

        let badGate = try Self.capabilitiesFromDescriptorMutation("qwen35_text.cam") { object in
            var exports = try #require(object["exports"] as? [[String: Any]])
            exports[0]["gates"] = ["qwen_tts_startup"]
            object["exports"] = exports
            var gates = try #require(object["gateContracts"] as? [[String: Any]])
            gates[0]["gateID"] = "qwen_tts_startup"
            object["gateContracts"] = gates
        }
        #expect(throws: SmeltCAMPackageCapabilitiesError.self) {
            _ = try badGate.runtimeAssemblyFeatureContract(for: badGate.resolve(.runText))
        }
    }

    @Test func releaseGateContractsPreserveEvidenceAndMultiRequirementGates() throws {
        let qwenTTS = try Self.releaseContracts(
            "qwen3_tts.cam",
            requiredGateIDs: ["startup", "audio_contract"],
            exportID: "synth",
            flowID: "synth"
        )
        let audioContract = try #require(qwenTTS["audio_contract"])
        #expect(audioContract.kind == "inventory")
        #expect(audioContract.gateRequirements == [
            "audio-rate:==:24khz:none",
            "package-files:include:manifest.json,weights.bin,model.metallib,trunk,trunk-mtp,vocab.json,merges.txt,tokenizer_config.json,config.json,module.json:none",
            "release-surface-ids:include:gate.startup-audio,gate.audio-contract,correctness.stream-parity,release.verify:none",
        ])
        #expect(audioContract.selectedOutputName == "audio")
        #expect(audioContract.selectedOutput == "pcm[dtype=f32,rate=24khz]")
    }

    @Test func releaseGateContractsProjectDS4HeavyQuantCanaryWithoutCatalogSupportClaim() throws {
        let ds4 = try Self.releaseContracts(
            "ds4_heavy_quant.cam",
            requiredGateIDs: ["startup", "quant_quality"],
            exportID: "generate",
            flowID: "generate"
        )
        let startup = try #require(ds4["startup"])
        #expect(startup.kind == "scalar-metric")
        #expect(startup.measurements == [])
        #expect(startup.fromEventID == "flow.accepted:generate")
        #expect(startup.toEventID == "emit:generate.text")
        #expect(startup.metricPath == "scalar-metric:gate.startup:elapsed")

        let quantQuality = try #require(ds4["quant_quality"])
        #expect(quantQuality.kind == "scalar-metric")
        #expect(quantQuality.gateRequirements == [
            "calibration.gptq.rank:>=:128:none",
            "perplexity.delta:<=:0.05:none",
        ])
        #expect(quantQuality.selectedOutputName == "text")
        #expect(quantQuality.contractSHA256 != startup.contractSHA256)
    }

    @Test func releaseGateContractsUseCanonicalPayloadHashesAndSurfaceOrder() throws {
        let capabilities = try Self.capabilities("qwen35_text.cam")
        let requiredGateIDs = ["startup", "prefill", "decode"]
        let contracts = try capabilities.releaseGateContracts(
            requiredGateIDs: requiredGateIDs,
            releaseSurfaces: [
                Self.releaseSurface(
                    exportID: "generate",
                    flowID: "generate",
                    gateIDs: requiredGateIDs
                ),
            ]
        )
        for contract in contracts {
            #expect(contract.schema == SmeltCAMReleaseGateContract.currentSchema)
            #expect(contract.contractID == "contract-\(contract.contractSHA256.prefix(16))")
            let payloadSHA256 = try Self.contractPayloadSHA256(contract)
            #expect(contract.contractSHA256 == payloadSHA256)
        }

        let reordered = Self.releaseSurface(
            exportID: "generate",
            flowID: "generate",
            gateIDs: ["decode", "startup"]
        )
        #expect(
            try capabilities.releaseContractIDs(for: reordered, contracts: contracts)
                == [
                    try #require(contracts.first { $0.gateID == "decode" }).contractID,
                    try #require(contracts.first { $0.gateID == "startup" }).contractID,
                ]
        )
    }

    @Test func releaseGateContractsIncludeDescriptorIdentityInPayloadHash() throws {
        let baseline = try Self.capabilities("qwen35_text.cam")
        let baselineContract = try #require(
            baseline.releaseGateContracts(
                requiredGateIDs: ["startup"],
                releaseSurfaces: [
                    Self.releaseSurface(
                        exportID: "generate",
                        flowID: "generate",
                        gateIDs: ["startup"]
                    ),
                ]
            ).first
        )
        #expect(baselineContract.camSemanticSHA256 == baseline.camSemanticSHA256)
        #expect(baselineContract.exportABISHA256 == baseline.exportABISHA256)
        #expect(
            baselineContract.descriptorGraphSignatureSHA256
                == baseline.descriptorGraphSignatureSHA256
        )

        let driftedGraphSignature = try Self.capabilitiesFromDescriptorMutation("qwen35_text.cam") {
            object in
            object["graphSignature"] = ["drifted graph signature line"]
        }
        let driftedContract = try #require(
            driftedGraphSignature.releaseGateContracts(
                requiredGateIDs: ["startup"],
                releaseSurfaces: [
                    Self.releaseSurface(
                        exportID: "generate",
                        flowID: "generate",
                        gateIDs: ["startup"]
                    ),
                ]
            ).first
        )
        #expect(
            driftedContract.descriptorGraphSignatureSHA256
                != baselineContract.descriptorGraphSignatureSHA256
        )
        #expect(driftedContract.contractSHA256 != baselineContract.contractSHA256)
    }

    @Test func releaseGateContractsRejectInvalidSurfaceBindings() throws {
        let capabilities = try Self.capabilities("qwen3_tts.cam")
        #expect(throws: SmeltCAMReleaseGateContractError.invalidSurfaceBinding(
            surfaceID: "release.verify",
            reason: "flow wrong does not match contract flow synth"
        )) {
            _ = try capabilities.releaseGateContracts(
                requiredGateIDs: ["startup"],
                releaseSurfaces: [
                    Self.releaseSurface(
                        exportID: "synth",
                        flowID: "wrong",
                        gateIDs: ["startup"]
                    ),
                ]
            )
        }
    }

    @Test func releaseGateContractsRejectSelectedInputDrift() throws {
        let capabilities = try Self.capabilities("qwen35_text.cam")
        let contracts = try capabilities.releaseGateContracts(
            requiredGateIDs: ["startup"],
            releaseSurfaces: [
                Self.releaseSurface(
                    exportID: "generate",
                    flowID: "generate",
                    gateIDs: ["startup"]
                ),
            ]
        )
        #expect(throws: SmeltCAMReleaseGateContractError.invalidSurfaceBinding(
            surfaceID: "release.verify",
            reason: "selected input names do not match contract inputs"
        )) {
            _ = try capabilities.releaseContractIDs(
                for: SmeltCAMReleaseSurfaceBinding(
                    surfaceID: "release.verify",
                    exportID: "generate",
                    flowID: "generate",
                    selectedInputNames: ["wrong"],
                    selectedInputs: ["text[encoding=utf8]"],
                    gateIDs: ["startup"],
                    requiresReleaseEvidence: true
                ),
                contracts: contracts
            )
        }
        #expect(throws: SmeltCAMReleaseGateContractError.invalidSurfaceBinding(
            surfaceID: "release.verify",
            reason: "selected inputs do not match contract inputs"
        )) {
            _ = try capabilities.releaseContractIDs(
                for: SmeltCAMReleaseSurfaceBinding(
                    surfaceID: "release.verify",
                    exportID: "generate",
                    flowID: "generate",
                    selectedInputNames: ["prompt"],
                    selectedInputs: ["audio"],
                    gateIDs: ["startup"],
                    requiresReleaseEvidence: true
                ),
                contracts: contracts
            )
        }
    }

    @Test func releaseGateContractsRejectUnboundReleaseEvidenceSurfaces() throws {
        let capabilities = try Self.capabilities("qwen35_text.cam")
        #expect(throws: SmeltCAMReleaseGateContractError.invalidSurfaceBinding(
            surfaceID: "release.verify",
            reason: "release evidence surface must name export and flow"
        )) {
            _ = try capabilities.releaseGateContracts(
                requiredGateIDs: ["startup"],
                releaseSurfaces: [
                    SmeltCAMReleaseSurfaceBinding(
                        surfaceID: "release.verify",
                        exportID: nil,
                        flowID: nil,
                        gateIDs: ["startup"],
                        requiresReleaseEvidence: true
                    ),
                ]
            )
        }
    }

    @Test func releaseGateContractsRejectDuplicateReleaseSurfaceBindings() throws {
        let capabilities = try Self.capabilities("qwen35_text.cam")
        #expect(throws: SmeltCAMReleaseGateContractError.invalidSurfaceBinding(
            surfaceID: "release.verify",
            reason: "release evidence surface repeats gate startup"
        )) {
            _ = try capabilities.releaseGateContracts(
                requiredGateIDs: ["startup"],
                releaseSurfaces: [
                    Self.releaseSurface(
                        exportID: "generate",
                        flowID: "generate",
                        gateIDs: ["startup", "startup"]
                    ),
                ]
            )
        }
        #expect(throws: SmeltCAMReleaseGateContractError.invalidSurfaceBinding(
            surfaceID: "release.verify",
            reason: "duplicate release evidence surface id"
        )) {
            _ = try capabilities.releaseGateContracts(
                requiredGateIDs: ["startup", "prefill"],
                releaseSurfaces: [
                    Self.releaseSurface(
                        exportID: "generate",
                        flowID: "generate",
                        gateIDs: ["startup"]
                    ),
                    Self.releaseSurface(
                        exportID: "generate",
                        flowID: "generate",
                        gateIDs: ["prefill"]
                    ),
                ]
            )
        }
    }

    @Test func releaseGateContractsRejectAmbiguousOutputFallback() throws {
        let capabilities = try Self.capabilitiesFromDescriptorMutation("qwen35_text.cam") {
            object in
            try Self.appendExportOutput(
                in: &object,
                exportID: "generate",
                portName: "alt",
                typeName: "text",
                attributes: ["encoding": "utf8"]
            )
        }

        #expect(throws: SmeltCAMReleaseGateContractError.ambiguousReleaseOutput(
            gateID: "prefill",
            outputs: ["alt", "text"]
        )) {
            _ = try capabilities.releaseGateContracts(
                requiredGateIDs: ["prefill"],
                releaseSurfaces: [
                    Self.releaseSurface(
                        exportID: "generate",
                        flowID: "generate",
                        gateIDs: ["prefill"]
                    ),
                ]
            )
        }
    }

    @Test func releaseGateContractsUseSurfaceBindingBeforeGateIDFallback() throws {
        let disambiguated = try Self.capabilitiesFromDescriptorMutation("qwen35_text.cam") {
            object in
            try Self.appendGenerateAltExport(in: &object, gates: ["startup"])
        }

        let contracts = try disambiguated.releaseGateContracts(
            requiredGateIDs: ["startup"],
            releaseSurfaces: [
                Self.releaseSurface(
                    exportID: "generate",
                    flowID: "generate",
                    gateIDs: ["startup"]
                ),
            ]
        )
        let contract = try #require(contracts.first)
        #expect(contract.exportID == "generate")
        #expect(contract.flowID == "generate")

        let invalid = try Self.capabilitiesFromDescriptorMutation("qwen35_text.cam") {
            object in
            try Self.appendGenerateAltExport(in: &object, gates: [])
        }
        #expect(throws: SmeltCAMReleaseGateContractError.invalidSurfaceBinding(
            surfaceID: "release.verify",
            reason: "export generate_alt does not expose gate startup"
        )) {
            _ = try invalid.releaseGateContracts(
                requiredGateIDs: ["startup"],
                releaseSurfaces: [
                    Self.releaseSurface(
                        exportID: "generate_alt",
                        flowID: "generate",
                        gateIDs: ["startup"]
                    ),
                ]
            )
        }
    }

    @Test func audioDeliveryRequiresStreamingExportFact() throws {
        var object = try Self.descriptorJSONObject("qwen3_tts.cam")
        var exports = try #require(object["exports"] as? [[String: Any]])
        exports[0]["capabilities"] = ["run.synthesize", "prepare.voice-defaults"]
        object["exports"] = exports
        let descriptor = try Self.decodeDescriptor(object)
        try descriptor.validateDecoded()

        let capabilities = try SmeltCAMPackageCapabilities(descriptor: descriptor)
        _ = try capabilities.resolve(.runAudio)
        _ = try capabilities.resolve(.traceTextSynthesize)
        _ = try capabilities.resolve(.prepareVoiceDefaults)
        #expect(throws: SmeltCAMPackageCapabilitiesError.self) {
            _ = try capabilities.resolve(.serveAudio)
        }
    }

    @Test func streamingFactAloneDoesNotSelectAudioOperation() throws {
        var object = try Self.descriptorJSONObject("qwen3_tts.cam")
        var exports = try #require(object["exports"] as? [[String: Any]])
        exports[0]["capabilities"] = ["run.stream"]
        object["exports"] = exports
        let descriptor = try Self.decodeDescriptor(object)
        try descriptor.validateDecoded()

        let capabilities = try SmeltCAMPackageCapabilities(descriptor: descriptor)
        #expect(throws: SmeltCAMPackageCapabilitiesError.self) {
            _ = try capabilities.resolve(.runAudio)
        }
        #expect(throws: SmeltCAMPackageCapabilitiesError.self) {
            _ = try capabilities.resolve(.traceTextSynthesize)
        }
        #expect(throws: SmeltCAMPackageCapabilitiesError.self) {
            _ = try capabilities.resolve(.serveAudio)
        }
    }

    @Test func requiredExportFactsCanNarrowAmbiguousAudioExports() throws {
        var object = try Self.descriptorJSONObject("qwen3_tts.cam")
        var exports = try #require(object["exports"] as? [[String: Any]])
        var bindings = try #require(object["exportFlowBindings"] as? [[String: Any]])
        var offlineExport = exports[0]
        offlineExport["exportID"] = "synth_offline"
        offlineExport["capabilities"] = ["run.synthesize"]
        exports.append(offlineExport)
        bindings.append(["exportID": "synth_offline", "flowID": "synth"])
        object["exports"] = exports
        object["exportFlowBindings"] = bindings
        let descriptor = try Self.decodeDescriptor(object)
        try descriptor.validateDecoded()

        let capabilities = try SmeltCAMPackageCapabilities(descriptor: descriptor)
        #expect(throws: SmeltCAMPackageCapabilitiesError.ambiguousExport(
            "run audio",
            ["synth", "synth_offline"]
        )) {
            _ = try capabilities.resolve(.runAudio)
        }
        let serve = try capabilities.resolve(.serveAudio)
        #expect(serve.exportID == "synth")
    }

    @Test func requiredExportFactsDoNotRankTwoMatchingAudioExports() throws {
        var object = try Self.descriptorJSONObject("qwen3_tts.cam")
        var exports = try #require(object["exports"] as? [[String: Any]])
        var bindings = try #require(object["exportFlowBindings"] as? [[String: Any]])
        var streamExport = exports[0]
        streamExport["exportID"] = "synth_stream_alt"
        exports.append(streamExport)
        bindings.append(["exportID": "synth_stream_alt", "flowID": "synth"])
        object["exports"] = exports
        object["exportFlowBindings"] = bindings
        let descriptor = try Self.decodeDescriptor(object)
        try descriptor.validateDecoded()

        let capabilities = try SmeltCAMPackageCapabilities(descriptor: descriptor)
        #expect(throws: SmeltCAMPackageCapabilitiesError.ambiguousExport(
            "serve audio",
            ["synth", "synth_stream_alt"]
        )) {
            _ = try capabilities.resolve(.serveAudio)
        }
    }

    @Test func topLevelDescriptorCapabilitiesDoNotSelectExport() throws {
        var object = try Self.descriptorJSONObject("qwen35_text.cam")
        var exports = try #require(object["exports"] as? [[String: Any]])
        exports[0]["capabilities"] = []
        object["exports"] = exports
        let descriptor = try Self.decodeDescriptor(object)
        #expect(Set(descriptor.capabilities) == ["prepare.prompt-prefix", "run.generate"])
        try descriptor.validateDecoded()

        let capabilities = try SmeltCAMPackageCapabilities(descriptor: descriptor)
        #expect(throws: SmeltCAMPackageCapabilitiesError.self) {
            _ = try capabilities.resolve(.runText)
        }
        #expect(throws: SmeltCAMPackageCapabilitiesError.self) {
            _ = try capabilities.resolve(.prepareTextPromptPrefix)
        }
    }

    @Test func runOnlyExportDoesNotSatisfyPreparationRequest() throws {
        var object = try Self.descriptorJSONObject("qwen35_text.cam")
        var exports = try #require(object["exports"] as? [[String: Any]])
        exports[0]["capabilities"] = ["run.generate"]
        object["exports"] = exports
        let descriptor = try Self.decodeDescriptor(object)
        try descriptor.validateDecoded()

        let capabilities = try SmeltCAMPackageCapabilities(descriptor: descriptor)
        _ = try capabilities.resolve(.runText)
        #expect(throws: SmeltCAMPackageCapabilitiesError.self) {
            _ = try capabilities.resolve(.prepareTextPromptPrefix)
        }
    }

    @Test func textRequestsRequireSingleUTF8TextInputAndOutput() throws {
        let inputDrift = try Self.capabilitiesFromDescriptorMutation("qwen35_text.cam") { object in
            var exports = try #require(object["exports"] as? [[String: Any]])
            try Self.mutatePortAttributes(
                in: &exports[0],
                portListKey: "inputs",
                index: 0,
                attributes: ["encoding": "utf16"]
            )
            object["exports"] = exports
            try Self.mutateGraphNodePortAttributes(
                in: &object,
                nodeID: "tokenizer",
                portListKey: "inputs",
                portName: "prompt",
                attributes: ["encoding": "utf16"]
            )
            try Self.mutateGraphEdgeValueType(
                in: &object,
                typeName: "text",
                attributes: ["encoding": "utf16"]
            ) { from, to in
                from["endpointType"] as? String == "moduleInput"
                    && from["name"] as? String == "prompt"
                    && to["endpointType"] as? String == "nodePort"
                    && to["nodeID"] as? String == "tokenizer"
                    && to["portName"] as? String == "prompt"
            }
        }
        for request in Self.textRequests {
            #expect(throws: SmeltCAMPackageCapabilitiesError.self) {
                _ = try inputDrift.resolve(request)
            }
        }

        let outputDrift = try Self.capabilitiesFromDescriptorMutation("qwen35_text.cam") { object in
            var exports = try #require(object["exports"] as? [[String: Any]])
            try Self.mutatePortAttributes(
                in: &exports[0],
                portListKey: "outputs",
                index: 0,
                attributes: [:]
            )
            object["exports"] = exports
        }
        for request in Self.textRequests {
            #expect(throws: SmeltCAMPackageCapabilitiesError.self) {
                _ = try outputDrift.resolve(request)
            }
        }

        let reasoner = try Self.capabilities("qwen35_reasoner.cam")
        #expect(throws: SmeltCAMPackageCapabilitiesError.self) {
            _ = try reasoner.resolve(.runText)
        }
    }

    @Test func exactTwoTextRequestsRequireExactlyTwoRequiredUTF8InputsAndOneOutput() throws {
        let oneInput = try Self.capabilitiesFromDescriptorMutation("qwen35_reasoner.cam") { object in
            var exports = try #require(object["exports"] as? [[String: Any]])
            var inputs = try #require(exports[0]["inputs"] as? [[String: Any]])
            inputs.removeLast()
            exports[0]["inputs"] = inputs
            object["exports"] = exports
            var edges = try #require(object["graphEdges"] as? [[String: Any]])
            edges.removeAll { edge in
                guard let from = edge["from"] as? [String: Any] else { return false }
                return from["endpointType"] as? String == "moduleInput"
                    && from["name"] as? String == "context"
            }
            object["graphEdges"] = edges
        }
        #expect(throws: SmeltCAMPackageCapabilitiesError.self) {
            _ = try oneInput.resolve(Self.twoTextRun)
        }

        let optionalInput = try Self.capabilitiesFromDescriptorMutation("qwen35_reasoner.cam") {
            object in
            var exports = try #require(object["exports"] as? [[String: Any]])
            var inputs = try #require(exports[0]["inputs"] as? [[String: Any]])
            inputs[1]["optional"] = true
            exports[0]["inputs"] = inputs
            object["exports"] = exports
        }
        #expect(throws: SmeltCAMPackageCapabilitiesError.self) {
            _ = try optionalInput.resolve(Self.twoTextRun)
        }

        let threeInputs = try Self.capabilitiesFromDescriptorMutation("qwen35_reasoner.cam") {
            object in
            var exports = try #require(object["exports"] as? [[String: Any]])
            var inputs = try #require(exports[0]["inputs"] as? [[String: Any]])
            var extra = inputs[0]
            extra["portName"] = "extra"
            inputs.append(extra)
            exports[0]["inputs"] = inputs
            object["exports"] = exports
        }
        #expect(throws: SmeltCAMPackageCapabilitiesError.self) {
            _ = try threeInputs.resolve(Self.twoTextRun)
        }

        let wrongEncoding = try Self.capabilitiesFromDescriptorMutation("qwen35_reasoner.cam") {
            object in
            var exports = try #require(object["exports"] as? [[String: Any]])
            try Self.mutatePortAttributes(
                in: &exports[0],
                portListKey: "inputs",
                index: 1,
                attributes: ["encoding": "utf16"]
            )
            object["exports"] = exports
            try Self.mutateGraphEdgeValueType(
                in: &object,
                typeName: "text",
                attributes: ["encoding": "utf16"]
            ) { from, _ in
                from["endpointType"] as? String == "moduleInput"
                    && from["name"] as? String == "context"
            }
            try Self.mutateGraphNodePortAttributes(
                in: &object,
                nodeID: "prompt_builder",
                portListKey: "inputs",
                portName: "context",
                attributes: ["encoding": "utf16"]
            )
        }
        #expect(throws: SmeltCAMPackageCapabilitiesError.self) {
            _ = try wrongEncoding.resolve(Self.twoTextRun)
        }

        let extraOutput = try Self.capabilitiesFromDescriptorMutation("qwen35_reasoner.cam") {
            object in
            try Self.appendExportOutput(
                in: &object,
                exportID: "review",
                portName: "debug_text",
                typeName: "text",
                attributes: ["encoding": "utf8"]
            )
        }
        #expect(throws: SmeltCAMPackageCapabilitiesError.self) {
            _ = try extraOutput.resolve(Self.twoTextRun)
        }
    }

    @Test func exactTwoTextGateObservationAndAmbiguityStayStrict() throws {
        let wrongGateFlow = try Self.capabilitiesFromDescriptorMutation("qwen35_reasoner.cam") {
            object in
            try Self.replaceGateToEndpoint(
                in: &object,
                gateID: "startup",
                endpoint: [
                    "endpointType": "nodePort",
                    "nodeID": "detokenizer",
                    "portName": "text",
                ]
            )
        }
        #expect(try wrongGateFlow.resolve(Self.twoTextRun).exportID == "review")
        #expect(throws: SmeltCAMPackageCapabilitiesError.self) {
            _ = try wrongGateFlow.resolve(Self.twoTextBench)
        }

        let missingTokenPredicate = try Self.capabilitiesFromDescriptorMutation(
            "qwen35_reasoner.cam"
        ) { object in
            try Self.replaceGateToPredicates(in: &object, gateID: "startup", predicates: [])
        }
        #expect(try missingTokenPredicate.resolve(Self.twoTextRun).exportID == "review")
        #expect(throws: SmeltCAMPackageCapabilitiesError.self) {
            _ = try missingTokenPredicate.resolve(Self.twoTextBench)
        }

        var object = try Self.descriptorJSONObject("qwen35_reasoner.cam")
        var exports = try #require(object["exports"] as? [[String: Any]])
        var bindings = try #require(object["exportFlowBindings"] as? [[String: Any]])
        var alternate = exports[0]
        alternate["exportID"] = "alternate"
        exports.append(alternate)
        bindings.append(["exportID": "alternate", "flowID": "review"])
        object["exports"] = exports
        object["exportFlowBindings"] = bindings
        let ambiguous = try SmeltCAMPackageCapabilities(descriptor: Self.decodeDescriptor(object))

        #expect(throws: SmeltCAMPackageCapabilitiesError.ambiguousExport(
            Self.twoTextRun.name,
            ["alternate", "review"]
        )) {
            _ = try ambiguous.resolve(Self.twoTextRun)
        }
    }

    @Test func runtimeAudioRejectsOutputAndFirstAudioGateFormatMismatch() throws {
        let wrongRate = try Self.capabilitiesFromDescriptorMutation("qwen3_tts.cam") { object in
            var exports = try #require(object["exports"] as? [[String: Any]])
            try Self.mutatePortAttributes(
                in: &exports[0],
                portListKey: "outputs",
                index: 0,
                attributes: ["dtype": "f32", "rate": "16khz"]
            )
            object["exports"] = exports
            try Self.mutateGraphNodePortAttributes(
                in: &object,
                nodeID: "codec-decoder",
                portListKey: "outputs",
                portName: "audio",
                attributes: ["dtype": "f32", "rate": "16khz"]
            )
            try Self.mutateGraphEdgeValueType(
                in: &object,
                typeName: "pcm",
                attributes: ["dtype": "f32", "rate": "16khz"]
            ) { from, to in
                from["endpointType"] as? String == "nodePort"
                    && from["nodeID"] as? String == "codec-decoder"
                    && from["portName"] as? String == "audio"
                    && to["endpointType"] as? String == "moduleOutput"
                    && to["name"] as? String == "audio"
            }
        }

        for request in Self.audioRequests {
            #expect(throws: SmeltCAMPackageCapabilitiesError.self) {
                _ = try wrongRate.resolve(request)
            }
        }
    }

    @Test func shapeFilteringDoesNotRankAmbiguity() throws {
        var object = try Self.descriptorJSONObject("qwen3_tts.cam")
        var exports = try #require(object["exports"] as? [[String: Any]])
        var bindings = try #require(object["exportFlowBindings"] as? [[String: Any]])
        var wrongShapeExport = exports[0]
        wrongShapeExport["exportID"] = "synth_wrong_rate"
        try Self.mutatePortAttributes(
            in: &wrongShapeExport,
            portListKey: "outputs",
            index: 0,
            attributes: ["dtype": "f32", "rate": "16khz"]
        )
        try Self.renamePort(in: &wrongShapeExport, portListKey: "outputs", index: 0, name: "audio_16khz")
        exports.append(wrongShapeExport)
        bindings.append(["exportID": "synth_wrong_rate", "flowID": "synth"])
        object["exports"] = exports
        object["exportFlowBindings"] = bindings
        let filtered = try SmeltCAMPackageCapabilities(descriptor: Self.decodeDescriptor(object))

        let run = try filtered.resolve(.runAudio)
        #expect(run.exportID == "synth")

        var validAlt = exports[0]
        validAlt["exportID"] = "synth_alt"
        exports.append(validAlt)
        bindings.append(["exportID": "synth_alt", "flowID": "synth"])
        object["exports"] = exports
        object["exportFlowBindings"] = bindings
        let ambiguous = try SmeltCAMPackageCapabilities(descriptor: Self.decodeDescriptor(object))

        #expect(throws: SmeltCAMPackageCapabilitiesError.ambiguousExport(
            "run audio",
            ["synth", "synth_alt"]
        )) {
            _ = try ambiguous.resolve(.runAudio)
        }
    }

    @Test func capabilityResolverSourceDoesNotNameCompletionTargets() throws {
        let source = try String(
            contentsOf: Self.repoRoot
                .appendingPathComponent("Sources", isDirectory: true)
                .appendingPathComponent("SmeltSchema", isDirectory: true)
                .appendingPathComponent("SmeltCAMPackageCapabilities.swift"),
            encoding: .utf8
        )
        let normalized = source.lowercased()
        for banned in [
            "qwen",
            "qwen3_tts",
            "reasoner",
            "review",
            "context",
            "candidate",
            "moduleid",
            "qwen35_reasoner.cam",
        ] {
            let leaked = normalized.contains(banned)
            #expect(!leaked, "\(banned) leaked into capability resolver")
        }
    }

    private static var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private static func descriptor(_ name: String) throws -> SmeltCAMPackageDescriptor {
        try SmeltCAMPackageDescriptor(from: registryModuleIR(name))
    }

    private static func descriptorData(_ name: String) throws -> Data {
        try descriptor(name).canonicalJSONData()
    }

    private static func capabilities(_ name: String) throws -> SmeltCAMPackageCapabilities {
        try SmeltCAMPackageCapabilities(descriptor: descriptor(name))
    }

    private static func runtimeAssemblyFeatureContract(
        _ name: String,
        _ request: SmeltCAMCapabilityRequest
    ) throws -> SmeltCAMPackageCapabilities.RuntimeAssemblyFeatureContract {
        let capabilities = try capabilities(name)
        return try capabilities.runtimeAssemblyFeatureContract(for: capabilities.resolve(request))
    }

    private static func releaseContracts(
        _ name: String,
        requiredGateIDs: [String],
        exportID: String,
        flowID: String
    ) throws -> [String: SmeltCAMReleaseGateContract] {
        let capabilities = try capabilities(name)
        let contracts = try capabilities.releaseGateContracts(
            requiredGateIDs: requiredGateIDs,
            releaseSurfaces: [
                releaseSurface(exportID: exportID, flowID: flowID, gateIDs: requiredGateIDs),
            ]
        )
        return Dictionary(uniqueKeysWithValues: contracts.map { ($0.gateID, $0) })
    }

    private static func releaseSurface(
        surfaceID: String = "release.verify",
        exportID: String,
        flowID: String,
        gateIDs: [String],
        requiresReleaseEvidence: Bool = true
    ) -> SmeltCAMReleaseSurfaceBinding {
        SmeltCAMReleaseSurfaceBinding(
            surfaceID: surfaceID,
            exportID: exportID,
            flowID: flowID,
            gateIDs: gateIDs,
            requiresReleaseEvidence: requiresReleaseEvidence
        )
    }

    private static let textRequests: [SmeltCAMCapabilityRequest] = [
        .runText,
        .benchDecode,
        .serveText,
        .prepareTextPromptPrefix,
        .traceTextGenerate,
    ]

    private static let twoTextRun = SmeltCAMCapabilityRequest.exactTextToText(
        name: "run text with two required text inputs",
        requiredTextInputCount: 2,
        requiredInputNames: ["candidate", "context"],
        requiredAnyExportFacts: ["run.generate"]
    )

    private static let twoTextBench = SmeltCAMCapabilityRequest.exactTextToText(
        name: "bench text with two required text inputs",
        requiredTextInputCount: 2,
        requiredInputNames: ["candidate", "context"],
        requiredAnyExportFacts: ["run.generate"],
        requiredGateObservations: [SmeltCAMCapabilityRequest.firstTextOutputObservation()]
    )

    private static let twoTextServe = SmeltCAMCapabilityRequest.exactTextToText(
        name: "serve text with two required text inputs",
        requiredTextInputCount: 2,
        requiredInputNames: ["candidate", "context"],
        requiredAnyExportFacts: ["run.generate"]
    )

    private static let twoTextTrace = SmeltCAMCapabilityRequest.exactTextToText(
        name: "trace text with two required text inputs",
        requiredTextInputCount: 2,
        requiredInputNames: ["candidate", "context"],
        requiredAnyExportFacts: ["run.generate"]
    )

    private static let twoTextPreparation = SmeltCAMCapabilityRequest.exactTextToText(
        name: "prepare prompt-prefix with two required text inputs",
        requiredTextInputCount: 2,
        requiredInputNames: ["candidate", "context"],
        requiredAnyExportFacts: ["prepare.prompt-prefix"]
    )

    private static let twoTextRequests: [SmeltCAMCapabilityRequest] = [
        twoTextRun,
        twoTextBench,
        twoTextServe,
        twoTextTrace,
        twoTextPreparation,
    ]

    private static let audioRequests: [SmeltCAMCapabilityRequest] = [
        .runAudio,
        .serveAudioStream,
        .serveAudio,
        .prepareVoiceDefaults,
        .traceAudioSynthesis,
        .traceTextSynthesize,
    ]

    private static let runtimeAudioRequests: [SmeltCAMCapabilityRequest] = [
        .runAudio,
        .serveAudioStream,
        .serveAudio,
        .traceAudioSynthesis,
        .traceTextSynthesize,
    ]

    private static func portShape(_ port: SmeltCAMPackageDescriptor.Port) -> String {
        guard !port.type.attributes.isEmpty else { return port.type.typeName }
        let attributes = port.type.attributes
            .map { "\($0.key)=\($0.value)" }
            .sorted()
            .joined(separator: ",")
        return "\(port.type.typeName)[\(attributes)]"
    }

    private static func namedPortShape(_ port: SmeltCAMPackageDescriptor.Port) -> String {
        "\(port.portName):\(portShape(port))"
    }

    private static func sha256Hex(_ value: String) -> String {
        SHA256.hash(data: Data(value.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    private static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func contractPayloadSHA256(
        _ contract: SmeltCAMReleaseGateContract
    ) throws -> String {
        let data = try contract.canonicalJSONData()
        var object = try #require(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        object.removeValue(forKey: "contract_id")
        object.removeValue(forKey: "contract_sha256")
        let payload = try JSONSerialization.data(
            withJSONObject: object,
            options: [.sortedKeys, .withoutEscapingSlashes]
        )
        return sha256Hex(payload)
    }

    private static func expectGraphAndSurfaceFeaturesAreSplit(
        _ contract: SmeltCAMPackageCapabilities.RuntimeAssemblyFeatureContract
    ) {
        #expect(contract.schema == SmeltCAMPackageCapabilities.RuntimeAssemblyFeatureContract.currentSchema)
        #expect(!contract.configuredGraphFeatureSet.isEmpty)
        #expect(!contract.featureSet.isEmpty)
        #expect(contract.configuredGraphFeatureSet == contract.configuredGraphFeatureSet.sorted())
        #expect(contract.featureSet == contract.featureSet.sorted())
        #expect(Set(contract.configuredGraphFeatureSet).count == contract.configuredGraphFeatureSet.count)
        #expect(Set(contract.featureSet).count == contract.featureSet.count)
        for feature in contract.configuredGraphFeatureSet {
            #expect(!feature.hasPrefix("io."), "\(feature)")
            #expect(!feature.hasPrefix("gate."), "\(feature)")
            #expect(contract.featureSet.contains(feature), "\(feature)")
        }
        #expect(contract.featureSet.contains { $0.hasPrefix("io.") })
        #expect(contract.featureSet.contains { $0.hasPrefix("gate.") })
    }

    private static let bannedFeatureTerms: Set<String> = [
        "qwen",
        "qwentts",
        "whisper",
        "deepseek",
        "ds4",
        "llm",
        "tts",
        "asr",
        "family",
        "kind",
        "arch",
        "architecture",
        "modality",
        "profile",
        "bucket",
        "target",
        "handler",
        "registry",
        "policy",
        "manifest",
        "bridge",
        "locator",
        "model",
        "modelname",
        "moduleid",
    ]

    private static func selectorFeatureViolations(_ feature: String) -> [String] {
        selectorTermCandidates(feature)
            .filter { bannedFeatureTerms.contains($0) }
            .sorted()
    }

    private static func selectorTermCandidates(_ value: String) -> Set<String> {
        let tokens = value
            .split { !$0.isLetter && !$0.isNumber }
            .map(String.init)
            .flatMap(splitCamelCaseToken)
            .map { $0.lowercased() }
            .filter { !$0.isEmpty }
        var candidates = Set(tokens)
        candidates.insert(tokens.joined())
        for index in tokens.indices.dropLast() {
            candidates.insert(tokens[index] + tokens[tokens.index(after: index)])
        }
        return candidates
    }

    private static func splitCamelCaseToken(_ value: String) -> [String] {
        var out: [String] = []
        var current = ""
        for character in value {
            if character.isUppercase && !current.isEmpty {
                out.append(current)
                current = ""
            }
            current.append(character)
        }
        if !current.isEmpty {
            out.append(current)
        }
        return out
    }

    private static func capabilitiesFromDescriptorMutation(
        _ name: String,
        mutate: (inout [String: Any]) throws -> Void
    ) throws -> SmeltCAMPackageCapabilities {
        var object = try descriptorJSONObject(name)
        try mutate(&object)
        return try SmeltCAMPackageCapabilities(descriptor: decodeDescriptor(object))
    }

    private static func expectRuntimeAudioRejectsButPreparationResolves(
        _ capabilities: SmeltCAMPackageCapabilities
    ) throws {
        for request in runtimeAudioRequests {
            #expect(throws: SmeltCAMPackageCapabilitiesError.noMatchingExport(request.name)) {
                _ = try capabilities.resolve(request)
            }
        }
        #expect(try capabilities.resolve(.prepareVoiceDefaults).exportID == "synth")
    }

    private static func mutatePortAttributes(
        in export: inout [String: Any],
        portListKey: String,
        index: Int,
        attributes: [String: String]
    ) throws {
        var ports = try #require(export[portListKey] as? [[String: Any]])
        var port = ports[index]
        var type = try #require(port["type"] as? [String: Any])
        type["attributes"] = attributes
        port["type"] = type
        ports[index] = port
        export[portListKey] = ports
    }

    private static func renamePort(
        in export: inout [String: Any],
        portListKey: String,
        index: Int,
        name: String
    ) throws {
        var ports = try #require(export[portListKey] as? [[String: Any]])
        ports[index]["portName"] = name
        export[portListKey] = ports
    }

    private static func mutateBlockCodecStreaming(
        in object: inout [String: Any],
        blockID: String,
        streaming: Bool
    ) throws {
        var blocks = try #require(object["blocks"] as? [[String: Any]])
        let index = try #require(blocks.indices.first {
            blocks[$0]["blockID"] as? String == blockID
        })
        var block = blocks[index]
        var shape = try #require(block["shape"] as? [String: Any])
        var codec = try #require(shape["codecDecoder"] as? [String: Any])
        codec["streaming"] = streaming
        shape["codecDecoder"] = codec
        block["shape"] = shape
        blocks[index] = block
        object["blocks"] = blocks
    }

    private static func removeBlockRequirement(
        in object: inout [String: Any],
        blockID: String,
        key: String
    ) throws {
        var blocks = try #require(object["blocks"] as? [[String: Any]])
        let index = try #require(blocks.indices.first {
            blocks[$0]["blockID"] as? String == blockID
        })
        var block = blocks[index]
        var shape = try #require(block["shape"] as? [String: Any])
        var requirements = try #require(shape["requirements"] as? [[String: Any]])
        requirements.removeAll { ($0["key"] as? String) == key }
        shape["requirements"] = requirements
        block["shape"] = shape
        blocks[index] = block
        object["blocks"] = blocks
    }

    private static func replaceTransformerLayerRoles(
        in object: inout [String: Any],
        blockID: String,
        roles: [String],
        removeDeltaShape: Bool = false
    ) throws {
        var blocks = try #require(object["blocks"] as? [[String: Any]])
        let index = try #require(blocks.indices.first {
            blocks[$0]["blockID"] as? String == blockID
        })
        var block = blocks[index]
        var shape = try #require(block["shape"] as? [String: Any])
        var transformer = try #require(shape["transformer"] as? [String: Any])
        var layers = try #require(transformer["layers"] as? [String: Any])
        layers["roles"] = roles
        transformer["layers"] = layers
        if removeDeltaShape {
            transformer.removeValue(forKey: "delta")
        }
        shape["transformer"] = transformer
        block["shape"] = shape
        blocks[index] = block
        object["blocks"] = blocks
    }

    private static func mutateGraphNodePortAttributes(
        in object: inout [String: Any],
        nodeID: String,
        portListKey: String,
        portName: String,
        attributes: [String: String]
    ) throws {
        var nodes = try #require(object["graphNodes"] as? [[String: Any]])
        let nodeIndex = try #require(nodes.indices.first { index in
            nodes[index]["nodeID"] as? String == nodeID
        })
        var node = nodes[nodeIndex]
        var ports = try #require(node[portListKey] as? [[String: Any]])
        let portIndex = try #require(ports.indices.first { index in
            ports[index]["portName"] as? String == portName
        })
        var port = ports[portIndex]
        var type = try #require(port["type"] as? [String: Any])
        type["attributes"] = attributes
        port["type"] = type
        ports[portIndex] = port
        node[portListKey] = ports
        nodes[nodeIndex] = node
        object["graphNodes"] = nodes
    }

    private static func mutateGraphEdgeValueType(
        in object: inout [String: Any],
        typeName: String,
        attributes: [String: String],
        where matches: ([String: Any], [String: Any]) -> Bool
    ) throws {
        var edges = try #require(object["graphEdges"] as? [[String: Any]])
        let edgeIndex = try #require(edges.indices.first { index in
            guard let from = edges[index]["from"] as? [String: Any],
                  let to = edges[index]["to"] as? [String: Any] else {
                return false
            }
            return matches(from, to)
        })
        edges[edgeIndex]["valueType"] = [
            "typeName": typeName,
            "attributes": attributes,
        ]
        object["graphEdges"] = edges
    }

    private static func setExportGates(
        in object: inout [String: Any],
        exportID: String,
        gates: [String]
    ) throws {
        var exports = try #require(object["exports"] as? [[String: Any]])
        let index = try #require(exports.indices.first { exports[$0]["exportID"] as? String == exportID })
        exports[index]["gates"] = gates
        object["exports"] = exports
    }

    private static func appendExportCapability(
        in object: inout [String: Any],
        exportID: String,
        capability: String
    ) throws {
        var exports = try #require(object["exports"] as? [[String: Any]])
        let index = try #require(exports.indices.first { exports[$0]["exportID"] as? String == exportID })
        var capabilities = try #require(exports[index]["capabilities"] as? [String])
        capabilities.append(capability)
        exports[index]["capabilities"] = capabilities
        object["exports"] = exports
    }

    private static func setCompileRequirementValue(
        in object: inout [String: Any],
        key: String,
        value: String
    ) throws {
        var compile = try #require(object["compileRequirements"] as? [[String: Any]])
        let index = try #require(
            compile.indices.first { (compile[$0]["key"] as? String) == key }
        )
        compile[index]["value"] = value
        object["compileRequirements"] = compile
    }

    private static func appendGenerateAltExport(
        in object: inout [String: Any],
        gates: [String]
    ) throws {
        var exports = try #require(object["exports"] as? [[String: Any]])
        var alt = try #require(exports.first { $0["exportID"] as? String == "generate" })
        alt["exportID"] = "generate_alt"
        alt["gates"] = gates
        exports.append(alt)
        object["exports"] = exports

        var bindings = try #require(object["exportFlowBindings"] as? [[String: Any]])
        bindings.append([
            "exportID": "generate_alt",
            "flowID": "generate",
        ])
        object["exportFlowBindings"] = bindings
    }

    private static func appendExportOutput(
        in object: inout [String: Any],
        exportID: String,
        portName: String,
        typeName: String,
        attributes: [String: String]
    ) throws {
        var exports = try #require(object["exports"] as? [[String: Any]])
        let index = try #require(exports.indices.first { exports[$0]["exportID"] as? String == exportID })
        var outputs = try #require(exports[index]["outputs"] as? [[String: Any]])
        outputs.append([
            "portName": portName,
            "optional": false,
            "type": ["typeName": typeName, "attributes": attributes],
        ])
        exports[index]["outputs"] = outputs
        object["exports"] = exports
    }

    private static func replaceGateRequirements(
        in object: inout [String: Any],
        gateID: String,
        requirements: [[String: Any]]
    ) throws {
        var gates = try #require(object["gateContracts"] as? [[String: Any]])
        let index = try #require(gates.indices.first { gates[$0]["gateID"] as? String == gateID })
        gates[index]["requirements"] = requirements
        let requirementSubjects = Set(requirements.compactMap { $0["subject"] as? String })
        let measurements = (gates[index]["measurements"] as? [[String: Any]]) ?? []
        gates[index]["measurements"] = measurements.filter {
            guard let subject = $0["subject"] as? String else { return false }
            return requirementSubjects.contains(subject)
        }
        object["gateContracts"] = gates
    }

    private static func replaceGateToEndpoint(
        in object: inout [String: Any],
        gateID: String,
        endpoint: [String: Any]
    ) throws {
        try updateGateTo(in: &object, gateID: gateID) { to in
            to["endpoint"] = endpoint
        }
    }

    private static func replaceGateTo(
        in object: inout [String: Any],
        gateID: String,
        to replacement: [String: Any]
    ) throws {
        var gates = try #require(object["gateContracts"] as? [[String: Any]])
        let index = try #require(gates.indices.first { gates[$0]["gateID"] as? String == gateID })
        gates[index]["to"] = replacement
        object["gateContracts"] = gates
    }

    private static func replaceGateFrom(
        in object: inout [String: Any],
        gateID: String,
        from replacement: [String: Any]
    ) throws {
        var gates = try #require(object["gateContracts"] as? [[String: Any]])
        let index = try #require(gates.indices.first { gates[$0]["gateID"] as? String == gateID })
        gates[index]["from"] = replacement
        object["gateContracts"] = gates
    }

    private static func replaceGateToPredicates(
        in object: inout [String: Any],
        gateID: String,
        predicates: [[String: Any]]
    ) throws {
        try updateGateTo(in: &object, gateID: gateID) { to in
            to["predicates"] = predicates
        }
    }

    private static func updateGateTo(
        in object: inout [String: Any],
        gateID: String,
        mutate: (inout [String: Any]) throws -> Void
    ) throws {
        var gates = try #require(object["gateContracts"] as? [[String: Any]])
        let index = try #require(gates.indices.first { gates[$0]["gateID"] as? String == gateID })
        var gate = gates[index]
        var to = try #require(gate["to"] as? [String: Any])
        try mutate(&to)
        gate["to"] = to
        gates[index] = gate
        object["gateContracts"] = gates
    }

    private static func appendGateContract(
        in object: inout [String: Any],
        gate: [String: Any]
    ) throws {
        var gates = try #require(object["gateContracts"] as? [[String: Any]])
        var gate = gate
        if gate["measurements"] == nil {
            gate["measurements"] = []
        }
        gates.append(gate)
        object["gateContracts"] = gates
    }

    private static func descriptorJSONObject(_ name: String) throws -> [String: Any] {
        let object = try JSONSerialization.jsonObject(with: descriptorData(name))
        return try #require(object as? [String: Any])
    }

    private static func decodeDescriptor(_ object: [String: Any]) throws -> SmeltCAMPackageDescriptor {
        let data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        return try JSONDecoder().decode(SmeltCAMPackageDescriptor.self, from: data)
    }

    private static func comparisonSignature(
        _ comparison: SmeltCAMPackageDescriptor.Comparison
    ) -> String {
        "\(comparison.subject):\(comparison.relation):\(comparison.value):\(comparison.unit ?? "none")"
    }

}
