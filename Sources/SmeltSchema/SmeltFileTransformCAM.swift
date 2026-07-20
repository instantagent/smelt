/// Reusable authored CAM shape for a single-step artifact transformation.
public enum SmeltFileTransformCAM {
    /// Builds a typed module whose selected flow invokes one native entrypoint.
    public static func module(
        moduleID: String,
        exportID: String,
        entrypoint: String,
        inputName: String,
        inputMediaType: String,
        outputName: String,
        outputMediaType: String
    ) -> SmeltCAMIR {
        let inputType = SmeltCAMIR.TypeRef(
            "artifact",
            attributes: ["media-type": inputMediaType]
        )
        let outputType = SmeltCAMIR.TypeRef(
            "artifact",
            attributes: ["media-type": outputMediaType]
        )
        return SmeltCAMIR(
            module: .init(id: moduleID),
            exports: [
                .init(
                    id: exportID,
                    inputs: [.init(name: inputName, type: inputType)],
                    outputs: [.init(name: outputName, type: outputType)],
                    capabilities: ["run.transform"]
                ),
            ],
            exportBindings: [.init(export: exportID, flow: "transform")],
            blocks: [],
            graphNodes: [
                .init(
                    id: "transform",
                    implementation: .native,
                    inputs: [.init(name: inputName, type: inputType)],
                    outputs: [.init(name: outputName, type: outputType)]
                ),
            ],
            graphEdges: [
                .init(
                    from: .moduleInput(inputName),
                    to: .node("transform", inputName),
                    type: inputType
                ),
                .init(
                    from: .node("transform", outputName),
                    to: .moduleOutput(outputName),
                    type: outputType
                ),
            ],
            flows: [
                .init(
                    id: "transform",
                    phases: [
                        .init(
                            role: .setup,
                            calls: [.node("transform", entrypoint: entrypoint)]
                        ),
                    ],
                    emit: [.node("transform", outputName)],
                    stop: []
                ),
            ],
            capabilities: ["run.transform"]
        )
    }
}
