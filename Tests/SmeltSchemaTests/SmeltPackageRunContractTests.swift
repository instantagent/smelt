import Foundation
import Testing
@testable import SmeltSchema

@Suite struct SmeltPackageRunContractTests {
    @Test func fileTransformCAMResolvesTypedPortsAndEntrypoint() throws {
        let module = SmeltFileTransformCAM.module(
            moduleID: "fixture_copy",
            exportID: "transform",
            entrypoint: "artifact.copy",
            inputName: "source",
            inputMediaType: "application/octet-stream",
            outputName: "destination",
            outputMediaType: "application/octet-stream"
        )
        let descriptor = try SmeltCAMPackageDescriptor(from: module)
        let capabilities = try SmeltCAMPackageCapabilities(descriptor: descriptor)
        let decision = try capabilities.resolve(
            .runFileTransform(
                inputName: "source",
                inputMediaType: "application/octet-stream",
                outputName: "destination",
                outputMediaType: "application/octet-stream"
            )
        )
        let graph = try capabilities.executionGraph(for: decision)

        #expect(decision.exportID == "transform")
        #expect(decision.flowID == "transform")
        #expect(graph.nodes.count == 1)
        #expect(graph.phases.flatMap(\.calls).map(\.entrypoint) == ["artifact.copy"])
    }

    @Test func fileTransformContractRoundTripsAndValidates() throws {
        let contract = SmeltPackageRunContract(
            export: "transform",
            entrypoint: "fixture.copy",
            input: .init(
                flag: "input",
                mediaTypes: ["model/gltf-binary"],
                fileExtensions: ["glb"],
                help: "Input mesh"
            ),
            output: .init(
                flag: "output",
                mediaTypes: ["model/gltf-binary"],
                fileExtensions: ["glb"],
                help: "Output mesh"
            ),
            options: [
                .init(
                    flag: "seed",
                    value: .unsignedInteger,
                    defaultValue: "0",
                    help: "Deterministic seed"
                ),
            ]
        )

        try contract.validate()
        let encoded = try JSONEncoder().encode(contract)
        let decoded = try JSONDecoder().decode(SmeltPackageRunContract.self, from: encoded)
        #expect(decoded == contract)
    }

    @Test func duplicatedPortAndOptionFlagsAreRejected() {
        let contract = SmeltPackageRunContract(
            export: "transform",
            entrypoint: "fixture.copy",
            input: .init(
                flag: "file",
                mediaTypes: ["application/octet-stream"],
                fileExtensions: ["bin"],
                help: "Input"
            ),
            output: .init(
                flag: "output",
                mediaTypes: ["application/octet-stream"],
                fileExtensions: ["bin"],
                help: "Output"
            ),
            options: [
                .init(flag: "file", value: .string, help: "Collision"),
            ]
        )

        #expect(throws: SmeltPackageRunContractError.self) {
            try contract.validate()
        }
    }

    @Test func numericDefaultsAreTypeChecked() {
        let contract = SmeltPackageRunContract(
            export: "transform",
            entrypoint: "fixture.copy",
            input: .init(
                flag: "input",
                mediaTypes: ["application/octet-stream"],
                fileExtensions: ["bin"],
                help: "Input"
            ),
            output: .init(
                flag: "output",
                mediaTypes: ["application/octet-stream"],
                fileExtensions: ["bin"],
                help: "Output"
            ),
            options: [
                .init(
                    flag: "count",
                    value: .positiveInteger,
                    defaultValue: "0",
                    help: "Count"
                ),
            ]
        )

        #expect(throws: SmeltPackageRunContractError.self) {
            try contract.validate()
        }
    }
}
