import Foundation
import Testing
@testable import SmeltCompiler
@testable import SmeltSchema

@Suite struct SmeltCAMPackageDescriptorTests {
    private static let representativeExamples = [
        "qwen35_text.cam",
        "qwen35_fast.cam",
        "qwen35_reasoner.cam",
        "qwen3_tts.cam",
        "ds4_heavy_quant.cam",
    ]

    @Test func descriptorsAreDeterministicForRepresentativeExamples() throws {
        for name in Self.representativeExamples {
            let ir = registryModuleIR(name)
            let first = try SmeltCAMPackageDescriptor(from: ir)
            let second = try SmeltCAMPackageDescriptor(from: ir)

            #expect(first == second)
            #expect(try first.canonicalJSONData() == second.canonicalJSONData())
            #expect(first.descriptorSchema == SmeltCAMPackageDescriptor.currentDescriptorSchema)
            #expect(first.descriptorVersion == SmeltCAMPackageDescriptor.currentDescriptorVersion)
            #expect(first.camSchemaVersion == SmeltCAMIR.currentSchemaVersion)
            #expect(first.camSemanticSHA256 == (try ir.semanticSHA256()))
            #expect(first.exportABISHA256 == (try ir.exportABISHA256()))
            try first.validateDecoded()
            let decoded = try JSONDecoder().decode(
                SmeltCAMPackageDescriptor.self,
                from: first.canonicalJSONData()
            )
            try decoded.validateDecoded()
            try Self.assertDescriptorJSONHasNoRuntimeSelectorKeys(first, label: name)
        }
    }

    @Test func qwenTextDescriptorCarriesGraphFlowInventoryAndGateContracts() throws {
        let descriptor = try Self.descriptor("qwen35_text.cam")

        #expect(descriptor.moduleID == "qwen35_text")
        #expect(descriptor.exports.map(\.exportID) == ["generate"])
        #expect(descriptor.exportFlowBindings.map { "\($0.exportID):\($0.flowID)" } == [
            "generate:generate",
        ])
        #expect(descriptor.sourceReferences.map { "\($0.sourceID):\($0.sourceType)" } == [
            "tokenizer:hf-file",
            "weights:hf",
        ])
        let weights = try #require(
            descriptor.sourceReferences.first { $0.sourceID == "weights" }
        )
        #expect(weights.locator == "Qwen/Qwen3.5-2B")
        #expect(weights.revision == "main")
        #expect(weights.checkpointMap == "hf.qwen")
        let decoded = try JSONDecoder().decode(
            SmeltCAMPackageDescriptor.self,
            from: descriptor.canonicalJSONData()
        )
        let decodedWeights = try #require(
            decoded.sourceReferences.first { $0.sourceID == "weights" }
        )
        #expect(decodedWeights.checkpointMap == "hf.qwen")
        #expect(descriptor.blocks.map(\.blockID) == ["trunk"])
        #expect(descriptor.graphNodes.map(\.nodeID) == [
            "detokenizer", "sampler", "tokenizer", "trunk",
        ])
        let tokenizer = try #require(descriptor.graphNodes.first { $0.nodeID == "tokenizer" })
        #expect(tokenizer.annotations.map { "\($0.key)=\($0.value)" } == [
            "assistant-prelude=preclosed-think",
            "prompt-format=chatml",
            "tag=text-tokenizer",
            "thinking-policy=disabled",
            "tool-format=xml-function-parameters",
        ])
        #expect(!descriptor.compileRequirements.contains { $0.key == "package-spec" })
        #expect(descriptor.flows.first?.phases.map(\.phaseType) == ["setup", "step"])
        #expect(descriptor.flows.first?.stop.map(\.stopType).sorted() == [
            "eos-token", "eos-token", "host-cancel", "max-steps",
        ])
        #expect(descriptor.tensorBindings.map {
            "\($0.sourceID):\($0.tensorPattern.sourceID ?? ""):\($0.tensorPattern.pattern)->\($0.target.blockID).\($0.target.pattern):\($0.ownerBlockID)"
        } == [
            "weights:weights:*->trunk.*:trunk",
        ])
        #expect(descriptor.quantization.contains {
            $0.action == "default" && $0.storage?.storageFormat == "affine-u4"
                && $0.storage?.groupSize == 64
        })
        let inventory = try #require(
            descriptor.gateContracts.first { $0.gateID == "inventory" }?
                .requirements.first { $0.subject == "package-files" }
        )
        #expect(inventory.relation == "include")
        #expect(inventory.value.contains("prefill_dispatches.bin"))
    }

    @Test func qwenTTSAudioGateContractSurvivesDescriptorRoundTrip() throws {
        let descriptor = try Self.descriptor("qwen3_tts.cam")
        let decoded = try JSONDecoder().decode(
            SmeltCAMPackageDescriptor.self,
            from: descriptor.canonicalJSONData()
        )
        try decoded.validateDecoded()

        let startup = try #require(decoded.gateContracts.first { $0.gateID == "startup" })
        #expect(startup.from?.eventType == "flow.accepted")
        #expect(startup.from?.flowID == "synth")
        #expect(startup.to?.eventType == "emit")
        #expect(startup.to?.flowID == "synth")
        #expect(startup.to?.endpoint?.endpointType == "moduleOutput")
        #expect(startup.to?.endpoint?.name == "audio")
        #expect((startup.to?.predicates ?? []).map(Self.comparisonSignature) == [
            "duration:>=:20:ms",
            "format:==:pcm f32 24khz:none",
        ])
        #expect(startup.requirements.map(Self.comparisonSignature) == [
            "elapsed:<=:400:ms",
        ])
        #expect(startup.measurements.map {
            "\($0.subject):\($0.processMode):\($0.cacheState):\($0.occurrence)"
        } == [
            "elapsed:cold:cold:first",
        ])

        let audioContract = try #require(decoded.gateContracts.first { $0.gateID == "audio_contract" })
        let inventory = try #require(
            audioContract.requirements.first { $0.subject == "package-files" }
        )
        #expect(inventory.value.split(separator: ",").map(String.init).sorted() == [
            "config.json",
            "manifest.json",
            "merges.txt",
            "model.metallib",
            "module.json",
            "tokenizer_config.json",
            "trunk",
            "trunk-mtp",
            "vocab.json",
            "weights.bin",
        ])
    }

    @Test func ds4DescriptorKeepsPatternOnlyHeavyQuantContracts() throws {
        let descriptor = try Self.descriptor("ds4_heavy_quant.cam")

        #expect(descriptor.tensorBindings.map(\.target.pattern).sorted() == [
            "attention.*",
            "embeddings.*",
            "experts.*",
            "head.*",
            "router.*",
        ])
        #expect(descriptor.quantization.contains {
            $0.action == "default"
                && $0.storage?.storageFormat == "gptq"
                && $0.storage?.groupSize == 128
                && $0.calibration == nil
        })
        #expect(descriptor.quantization.contains {
            $0.action == "store"
                && $0.tensorPattern.pattern == "model.embed_tokens.*"
                && $0.storage?.storageFormat == "turbo-quant-h"
                && $0.calibration == nil
        })
        #expect(descriptor.quantization.contains {
            $0.action == "store"
                && $0.tensorPattern.pattern == "model.layers.*.mlp.experts.*"
                && $0.storage?.storageFormat == "gptq"
                && $0.calibration?.method == "gptq"
                && $0.calibration?.captures == ["trunk.attention", "trunk.experts"]
                && $0.calibration?.layersPerPass == 1
        })
        #expect(descriptor.quantization.filter { $0.calibration != nil }.map(\.tensorPattern.pattern) == [
            "model.layers.*.mlp.experts.*",
        ])
    }

    @Test func descriptorDoesNotProbeSources() throws {
        // Descriptor construction reads source locators without touching the
        // filesystem: a source pointing at a non-existent file still projects.
        let ir = try mutatedModuleIR(registryModuleIR("qwen35_text")) { object in
            var sources = object["sources"] as? [[String: Any]] ?? []
            sources.append([
                "id": "local_weights",
                "kind": "file",
                "locator": "/tmp/smelt/does-not-exist.safetensors",
            ])
            object["sources"] = sources
        }

        let descriptor = try SmeltCAMPackageDescriptor(from: ir)
        #expect(descriptor.moduleID == "qwen35_text")
        #expect(descriptor.sourceReferences.map(\.locator).contains("/tmp/smelt/does-not-exist.safetensors"))
    }

    @Test func descriptorChangesWhenFlowOrQuantContractsChange() throws {
        let base = registryModuleIR("qwen35_text")
        let baseline = try SmeltCAMPackageDescriptor(from: base)
        let stepDrift = try SmeltCAMPackageDescriptor(from: mutatedModuleIR(base) { object in
            guard var flows = object["flows"] as? [[String: Any]] else { return }
            var stop = flows[0]["stop"] as? [[String: Any]] ?? []
            if let index = stop.firstIndex(where: { $0["kind"] as? String == "max-steps" }) {
                stop[index]["value"] = 256
            }
            flows[0]["stop"] = stop
            object["flows"] = flows
        })
        let groupDrift = try SmeltCAMPackageDescriptor(from: mutatedModuleIR(base) { object in
            guard var quant = object["quantization"] as? [[String: Any]] else { return }
            if let index = quant.firstIndex(where: { $0["action"] as? String == "default" }) {
                var storage = quant[index]["storage"] as? [String: Any] ?? [:]
                storage["groupSize"] = 32
                quant[index]["storage"] = storage
            }
            object["quantization"] = quant
        })

        #expect(try baseline.canonicalJSONData() != stepDrift.canonicalJSONData())
        #expect(try baseline.canonicalJSONData() != groupDrift.canonicalJSONData())
        #expect(baseline.camSemanticSHA256 != stepDrift.camSemanticSHA256)
        #expect(baseline.camSemanticSHA256 != groupDrift.camSemanticSHA256)
    }

    @Test func promptPolicyChangesSemanticDescriptorButNotABIOrGraphSignature() throws {
        let baselineIR = registryModuleIR("qwen35_text")
        let changedIR = try mutatedModuleIR(baselineIR) { object in
            guard var nodes = object["graphNodes"] as? [[String: Any]],
                  let nodeIndex = nodes.firstIndex(where: { $0["id"] as? String == "tokenizer" })
            else { return }
            var annotations = nodes[nodeIndex]["annotations"] as? [[String: Any]] ?? []
            if let index = annotations.firstIndex(where: { $0["key"] as? String == "prompt-format" }) {
                annotations[index]["value"] = "raw"
            }
            nodes[nodeIndex]["annotations"] = annotations
            object["graphNodes"] = nodes
        }
        let baseline = try SmeltCAMPackageDescriptor(from: baselineIR)
        let mutated = try SmeltCAMPackageDescriptor(from: changedIR)

        #expect(baseline.camSemanticSHA256 != mutated.camSemanticSHA256)
        #expect(try baseline.canonicalJSONData() != mutated.canonicalJSONData())
        #expect(baseline.exportABISHA256 == mutated.exportABISHA256)
        #expect(baseline.graphSignature == mutated.graphSignature)
    }

    @Test func decodedDescriptorRejectsUnsupportedSchemaAndVersions() throws {
        try Self.expectDecodedValidationFailure(
            "schema",
            contains: "unsupported package descriptor schema"
        ) { object in
            object["descriptorSchema"] = "smelt.module.package_descriptor.v999"
        }
        try Self.expectDecodedValidationFailure(
            "descriptor version",
            contains: "unsupported package descriptor version"
        ) { object in
            object["descriptorVersion"] = 999
        }
        try Self.expectDecodedValidationFailure(
            "cam schema version",
            contains: "unsupported CAM schema version"
        ) { object in
            object["camSchemaVersion"] = 999
        }
    }

    @Test func decodedDescriptorRejectsDuplicateIdsAndBrokenBindings() throws {
        try Self.expectDecodedValidationFailure(
            "duplicate export",
            contains: "duplicate or empty export id"
        ) { object in
            var exports = try #require(object["exports"] as? [[String: Any]])
            exports.append(try #require(exports.first))
            object["exports"] = exports
        }
        try Self.expectDecodedValidationFailure(
            "missing flow binding",
            contains: "has no flow binding"
        ) { object in
            object["exportFlowBindings"] = []
        }
        try Self.expectDecodedValidationFailure(
            "unknown flow binding",
            contains: "export binding references unknown flow"
        ) { object in
            var bindings = try #require(object["exportFlowBindings"] as? [[String: Any]])
            bindings[0]["flowID"] = "missing_flow"
            object["exportFlowBindings"] = bindings
        }
    }

    @Test func decodedDescriptorRejectsBrokenGateAndEndpointReferences() throws {
        try Self.expectDecodedValidationFailure(
            "unknown export gate",
            contains: "references unknown gate"
        ) { object in
            var exports = try #require(object["exports"] as? [[String: Any]])
            var gates = try #require(exports[0]["gates"] as? [String])
            gates.append("missing_gate")
            exports[0]["gates"] = gates
            object["exports"] = exports
        }
        try Self.expectDecodedValidationFailure(
            "unknown flow call node",
            contains: "flow call references unknown graph node"
        ) { object in
            var flows = try #require(object["flows"] as? [[String: Any]])
            var phases = try #require(flows[0]["phases"] as? [[String: Any]])
            var calls = try #require(phases[0]["calls"] as? [[String: Any]])
            calls[0]["nodeID"] = "missing_node"
            phases[0]["calls"] = calls
            flows[0]["phases"] = phases
            object["flows"] = flows
        }
        try Self.expectDecodedValidationFailure(
            "unknown emit endpoint",
            contains: "malformed endpoint"
        ) { object in
            var flows = try #require(object["flows"] as? [[String: Any]])
            var emit = try #require(flows[0]["emit"] as? [[String: Any]])
            emit[0]["name"] = "missing_output"
            flows[0]["emit"] = emit
            object["flows"] = flows
        }
    }

    @Test func decodedDescriptorRejectsGraphEdgeValueTypeDrift() throws {
        try Self.expectDecodedValidationFailure(
            "local graph edge value type drift",
            contains: "declares value type pcm"
        ) { object in
            var edges = try #require(object["graphEdges"] as? [[String: Any]])
            let edgeIndex = try #require(edges.indices.first { index in
                guard let from = edges[index]["from"] as? [String: Any],
                      let to = edges[index]["to"] as? [String: Any] else {
                    return false
                }
                return from["endpointType"] as? String == "moduleInput"
                    && from["name"] as? String == "prompt"
                    && to["endpointType"] as? String == "nodePort"
                    && to["nodeID"] as? String == "tokenizer"
                    && to["portName"] as? String == "prompt"
            })
            edges[edgeIndex]["valueType"] = Self.wrongPCMValueType()
            object["graphEdges"] = edges
        }
    }

    @Test func decodedDescriptorRejectsImportedGraphNodeABIDrift() throws {
        let baseline = try Self.importedNodeDescriptorJSONObject()
        try Self.decodeDescriptor(baseline).validateDecoded()

        try Self.expectDecodedValidationFailure(
            "imported node missing optional input",
            object: baseline,
            contains: "missing input port 'speaker'"
        ) { object in
            try Self.removeGraphNodePort(
                in: &object,
                nodeID: "voice_node",
                portListKey: "inputs",
                portName: "speaker"
            )
        }

        try Self.expectDecodedValidationFailure(
            "imported node extra input",
            object: baseline,
            contains: "extra input port 'style'"
        ) { object in
            try Self.appendGraphNodePort(
                in: &object,
                nodeID: "voice_node",
                portListKey: "inputs",
                port: [
                    "portName": "style",
                    "optional": true,
                    "type": ["typeName": "voice-style", "attributes": [:]],
                ]
            )
        }

        try Self.expectDecodedValidationFailure(
            "imported node optional drift",
            object: baseline,
            contains: "expected optional=true, got optional=false"
        ) { object in
            try Self.setGraphNodePortOptional(
                in: &object,
                nodeID: "voice_node",
                portListKey: "inputs",
                portName: "speaker",
                optional: false
            )
        }

        try Self.expectDecodedValidationFailure(
            "imported node output type drift",
            object: baseline,
            contains: "expected pcm[dtype=f32,rate=24khz], got pcm[dtype=f32,rate=16khz]"
        ) { object in
            try Self.mutateGraphNodePortType(
                in: &object,
                nodeID: "voice_node",
                portListKey: "outputs",
                portName: "audio",
                typeName: "pcm",
                attributes: ["dtype": "f32", "rate": "16khz"]
            )
        }

        try Self.expectDecodedValidationFailure(
            "imported ABI hash drift",
            object: baseline,
            contains: "export ABI hash mismatch"
        ) { object in
            try Self.mutateImportedExportPortType(
                in: &object,
                alias: "voice",
                exportID: "synth",
                portListKey: "outputs",
                portName: "audio",
                typeName: "pcm",
                attributes: ["dtype": "f32", "rate": "16khz"]
            )
            try Self.mutateGraphNodePortType(
                in: &object,
                nodeID: "voice_node",
                portListKey: "outputs",
                portName: "audio",
                typeName: "pcm",
                attributes: ["dtype": "f32", "rate": "16khz"]
            )
        }
    }

    private static func descriptor(_ name: String) throws -> SmeltCAMPackageDescriptor {
        try SmeltCAMPackageDescriptor(from: registryModuleIR(name))
    }

    private static func descriptorJSONObject(_ name: String = "qwen35_text.cam") throws -> [String: Any] {
        let descriptor = try Self.descriptor(name)
        let object = try JSONSerialization.jsonObject(with: descriptor.canonicalJSONData())
        return try #require(object as? [String: Any])
    }

    private static func importedNodeDescriptorJSONObject() throws -> [String: Any] {
        let descriptor = try SmeltCAMPackageDescriptor(from: importedNodeIR())
        let object = try JSONSerialization.jsonObject(with: descriptor.canonicalJSONData())
        return try #require(object as? [String: Any])
    }

    private static func expectDecodedValidationFailure(
        _ label: String,
        in name: String = "qwen35_text.cam",
        contains expected: String,
        mutate: (inout [String: Any]) throws -> Void
    ) throws {
        let object = try descriptorJSONObject(name)
        try expectDecodedValidationFailure(label, object: object, contains: expected, mutate: mutate)
    }

    private static func expectDecodedValidationFailure(
        _ label: String,
        object original: [String: Any],
        contains expected: String,
        mutate: (inout [String: Any]) throws -> Void
    ) throws {
        var object = original
        try mutate(&object)
        let data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        let descriptor = try JSONDecoder().decode(SmeltCAMPackageDescriptor.self, from: data)
        do {
            try descriptor.validateDecoded()
            #expect(Bool(false), "\(label): expected decoded descriptor validation failure")
        } catch let error as SmeltCAMIRError {
            #expect(
                error.description.contains(expected),
                "\(label): expected '\(expected)', got '\(error.description)'"
            )
        } catch {
            #expect(Bool(false), "\(label): unexpected error \(error)")
        }
    }

    private static func decodeDescriptor(_ object: [String: Any]) throws -> SmeltCAMPackageDescriptor {
        let data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        return try JSONDecoder().decode(SmeltCAMPackageDescriptor.self, from: data)
    }

    private static func importedNodeIR() throws -> SmeltCAMIR {
        SmeltCAMIR(
            module: .init(id: "imported_node_parent"),
            imports: [speakerVoiceImport],
            exports: [.init(id: "run", inputs: [], outputs: [])],
            exportBindings: [.init(export: "run", flow: "run")],
            blocks: [],
            graphNodes: [
                .init(
                    id: "voice_node",
                    implementation: .imported,
                    imported: .init(alias: "voice", export: "synth"),
                    inputs: [speakerPort, textPort],
                    outputs: [audio24Port]
                ),
            ],
            graphEdges: [],
            flows: [
                .init(
                    id: "run",
                    phases: [],
                    emit: [],
                    stop: [.init(kind: .hostCancel)]
                ),
            ]
        )
    }

    private static func removeGraphNodePort(
        in object: inout [String: Any],
        nodeID: String,
        portListKey: String,
        portName: String
    ) throws {
        try updateGraphNodePortList(in: &object, nodeID: nodeID, portListKey: portListKey) { ports in
            ports.removeAll { $0["portName"] as? String == portName }
        }
    }

    private static func appendGraphNodePort(
        in object: inout [String: Any],
        nodeID: String,
        portListKey: String,
        port: [String: Any]
    ) throws {
        try updateGraphNodePortList(in: &object, nodeID: nodeID, portListKey: portListKey) { ports in
            ports.append(port)
        }
    }

    private static func setGraphNodePortOptional(
        in object: inout [String: Any],
        nodeID: String,
        portListKey: String,
        portName: String,
        optional: Bool
    ) throws {
        try updateGraphNodePort(in: &object, nodeID: nodeID, portListKey: portListKey, portName: portName) {
            $0["optional"] = optional
        }
    }

    private static func mutateGraphNodePortType(
        in object: inout [String: Any],
        nodeID: String,
        portListKey: String,
        portName: String,
        typeName: String,
        attributes: [String: String]
    ) throws {
        try updateGraphNodePort(in: &object, nodeID: nodeID, portListKey: portListKey, portName: portName) {
            $0["type"] = ["typeName": typeName, "attributes": attributes]
        }
    }

    private static func mutateImportedExportPortType(
        in object: inout [String: Any],
        alias: String,
        exportID: String,
        portListKey: String,
        portName: String,
        typeName: String,
        attributes: [String: String]
    ) throws {
        var imports = try #require(object["imports"] as? [[String: Any]])
        let importIndex = try #require(imports.indices.first { imports[$0]["alias"] as? String == alias })
        var imported = imports[importIndex]
        var exports = try #require(imported["exportABI"] as? [[String: Any]])
        let exportIndex = try #require(exports.indices.first { exports[$0]["exportID"] as? String == exportID })
        var export = exports[exportIndex]
        var ports = try #require(export[portListKey] as? [[String: Any]])
        let portIndex = try #require(ports.indices.first { ports[$0]["portName"] as? String == portName })
        ports[portIndex]["type"] = ["typeName": typeName, "attributes": attributes]
        export[portListKey] = ports
        exports[exportIndex] = export
        imported["exportABI"] = exports
        imports[importIndex] = imported
        object["imports"] = imports
    }

    private static func updateGraphNodePort(
        in object: inout [String: Any],
        nodeID: String,
        portListKey: String,
        portName: String,
        mutate: (inout [String: Any]) throws -> Void
    ) throws {
        try updateGraphNodePortList(in: &object, nodeID: nodeID, portListKey: portListKey) { ports in
            let portIndex = try #require(ports.indices.first { ports[$0]["portName"] as? String == portName })
            try mutate(&ports[portIndex])
        }
    }

    private static func updateGraphNodePortList(
        in object: inout [String: Any],
        nodeID: String,
        portListKey: String,
        mutate: (inout [[String: Any]]) throws -> Void
    ) throws {
        var nodes = try #require(object["graphNodes"] as? [[String: Any]])
        let nodeIndex = try #require(nodes.indices.first { nodes[$0]["nodeID"] as? String == nodeID })
        var node = nodes[nodeIndex]
        var ports = try #require(node[portListKey] as? [[String: Any]])
        try mutate(&ports)
        node[portListKey] = ports
        nodes[nodeIndex] = node
        object["graphNodes"] = nodes
    }

    private static let textPort = SmeltCAMIR.Port(
        name: "text",
        type: .init("text", attributes: ["encoding": "utf8"])
    )

    private static let speakerPort = SmeltCAMIR.Port(
        name: "speaker",
        type: .init("voice-id"),
        optional: true
    )

    private static let audio24Port = SmeltCAMIR.Port(
        name: "audio",
        type: .init("pcm", attributes: ["dtype": "f32", "rate": "24khz"])
    )

    private static let speakerSynthExport = SmeltCAMIR.Export(
        id: "synth",
        inputs: [textPort, speakerPort],
        outputs: [audio24Port],
        capabilities: ["run.synthesize", "run.stream"]
    )

    private static let speakerVoiceImport = SmeltCAMIR.Import(
        alias: "voice",
        moduleID: "qwen3_tts",
        irSHA256: String(repeating: "3", count: 64),
        exportABISHA256: try! SmeltCAMIR.exportABISHA256(for: [speakerSynthExport]),
        exportABI: [speakerSynthExport]
    )

    private static func wrongPCMValueType() -> [String: Any] {
        [
            "typeName": "pcm",
            "attributes": [
                "dtype": "f32",
                "rate": "24khz",
            ],
        ]
    }

    private static func assertDescriptorJSONHasNoRuntimeSelectorKeys(
        _ descriptor: SmeltCAMPackageDescriptor,
        label: String
    ) throws {
        let object = try JSONSerialization.jsonObject(with: descriptor.canonicalJSONData())
        let keys = Self.jsonKeys(object)
        let forbidden = [
            "architecture",
            "architectureclass",
            "domain",
            "enginefamily",
            "family",
            "familyname",
            "handler",
            "handlerkey",
            "kind",
            "llm",
            "modality",
            "modelprofile",
            "policymode",
            "routeselector",
            "routeset",
            "runtime",
            "runtimemode",
            "selector",
            "task",
            "tts",
            "asr",
        ]
        for key in keys {
            let normalized = key.lowercased()
            #expect(!forbidden.contains { normalized.contains($0) }, "\(label): forbidden descriptor key \(key)")
        }
    }

    private static func comparisonSignature(
        _ comparison: SmeltCAMPackageDescriptor.Comparison
    ) -> String {
        "\(comparison.subject):\(comparison.relation):\(comparison.value):\(comparison.unit ?? "none")"
    }

    private static func jsonKeys(_ object: Any) -> [String] {
        if let dictionary = object as? [String: Any] {
            return dictionary.flatMap { key, value in [key] + jsonKeys(value) }
        }
        if let array = object as? [Any] {
            return array.flatMap(jsonKeys)
        }
        return []
    }
}
