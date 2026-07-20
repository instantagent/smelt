import Foundation
import Testing

@testable import SmeltCompiler
@testable import SmeltRuntime
@testable import SmeltSchema

private struct SyntheticSafeTensor {
    let name: String
    let dtype: String
    let shape: [Int]
    let bytes: Data
}

private func writeSyntheticSafetensors(_ tensors: [SyntheticSafeTensor]) throws -> String {
    var offset = 0
    var header: [String: Any] = ["__metadata__": ["format": "mlx"]]
    var payload = Data()
    for tensor in tensors {
        header[tensor.name] = [
            "dtype": tensor.dtype,
            "shape": tensor.shape,
            "data_offsets": [offset, offset + tensor.bytes.count],
        ]
        payload.append(tensor.bytes)
        offset += tensor.bytes.count
    }
    let headerData = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
    var file = Data()
    var headerSize = UInt64(headerData.count).littleEndian
    withUnsafeBytes(of: &headerSize) { file.append(contentsOf: $0) }
    file.append(headerData)
    file.append(payload)
    let path = NSTemporaryDirectory()
        + "prequantized_\(ProcessInfo.processInfo.globallyUniqueString).safetensors"
    try file.write(to: URL(fileURLWithPath: path))
    return path
}

private func u16Data(_ values: [Float16]) -> Data {
    var bits = values.map(\.bitPattern)
    return bits.withUnsafeMutableBytes { Data($0) }
}

private func f32Data(_ values: [Float]) -> Data {
    var values = values
    return values.withUnsafeMutableBytes { Data($0) }
}

@Test func mlxTripletsBecomeOneSemanticBinaryTensor() throws {
    let path = try writeSyntheticSafetensors([
        .init(name: "model.proj.weight", dtype: "U32", shape: [2, 1],
              bytes: Data([0x55, 0x55, 0x55, 0x55, 0xAA, 0xAA, 0xAA, 0xAA])),
        .init(name: "model.proj.scales", dtype: "F16", shape: [2, 1],
              bytes: u16Data([0.5, 0.25])),
        .init(name: "model.proj.biases", dtype: "F16", shape: [2, 1],
              bytes: u16Data([-0.25, -0.125])),
        .init(name: "model.norm.weight", dtype: "F16", shape: [2],
              bytes: u16Data([1, 1])),
    ])
    defer { try? FileManager.default.removeItem(atPath: path) }

    let source = try SmeltPrequantizedSafetensors(
        loader: SafetensorsLoader(paths: [path]), format: .binary1, groupSize: 32)
    #expect(source.prequantizedTensors.count == 1)
    let view = try #require(source.prequantizedTensors.first)
    #expect(view.descriptor.sourceName == "model.proj.weight")
    #expect(view.descriptor.logicalShape == [2, 32])
    #expect(view.descriptor.packedRowStride == 4)
    #expect(view.descriptor.scaleCount == 2)
    #expect(Float16(bitPattern: try view.canonicalScaleBits(at: 0)) == 0.25)
    #expect(source.checkpointTensors.map(\.name) == ["model.norm.weight"])
    #expect(source.consumedTensorNames == [
        "model.proj.weight", "model.proj.scales", "model.proj.biases",
    ])
    #expect(source.normWeightSemantics == .directWeight)
    #expect(SmeltCompiler.shouldShiftNormWeightForCompatibility(
        runtimeName: "layers_0_input_layernorm_weight",
        config: SmeltModelIR.qwen35_2B.config,
        sourceSemantics: source.normWeightSemantics
    ))
    #expect(!SmeltCompiler.shouldShiftNormWeightForCompatibility(
        runtimeName: "layers_0_input_layernorm_weight",
        config: SmeltModelIR.qwen35_2B.config
    ))
}

@Test func mlxTripletsValidateTernaryCodesAndAffineRelationship() throws {
    let validPath = try writeSyntheticSafetensors([
        .init(name: "proj.weight", dtype: "U32", shape: [1, 1],
              bytes: Data([0b10_01_00_10, 0x55, 0x55, 0x55])),
        .init(name: "proj.scales", dtype: "F16", shape: [1, 1],
              bytes: u16Data([0.25])),
        .init(name: "proj.biases", dtype: "F16", shape: [1, 1],
              bytes: u16Data([-0.25])),
    ])
    defer { try? FileManager.default.removeItem(atPath: validPath) }
    let valid = try SmeltPrequantizedSafetensors(
        loader: SafetensorsLoader(paths: [validPath]), format: .ternary2, groupSize: 16)
    #expect(valid.prequantizedTensors.first?.descriptor.logicalShape == [1, 16])

    let invalidCodePath = try writeSyntheticSafetensors([
        .init(name: "proj.weight", dtype: "U32", shape: [1, 1],
              bytes: Data([0b11, 0x55, 0x55, 0x55])),
        .init(name: "proj.scales", dtype: "F16", shape: [1, 1],
              bytes: u16Data([0.25])),
        .init(name: "proj.biases", dtype: "F16", shape: [1, 1],
              bytes: u16Data([-0.25])),
    ])
    defer { try? FileManager.default.removeItem(atPath: invalidCodePath) }
    #expect(throws: SmeltPrequantizedTensorError.self) {
        try SmeltPrequantizedSafetensors(
            loader: SafetensorsLoader(paths: [invalidCodePath]),
            format: .ternary2, groupSize: 16)
    }

    let badBiasPath = try writeSyntheticSafetensors([
        .init(name: "proj.weight", dtype: "U32", shape: [1, 1],
              bytes: Data([0x55, 0x55, 0x55, 0x55])),
        .init(name: "proj.scales", dtype: "F16", shape: [1, 1],
              bytes: u16Data([0.25])),
        .init(name: "proj.biases", dtype: "F16", shape: [1, 1],
              bytes: u16Data([-0.5])),
    ])
    defer { try? FileManager.default.removeItem(atPath: badBiasPath) }
    #expect(throws: SmeltPrequantizedTensorError.self) {
        try SmeltPrequantizedSafetensors(
            loader: SafetensorsLoader(paths: [badBiasPath]),
            format: .ternary2, groupSize: 16)
    }
}

@Test func mlxTripletsRejectMissingAndOrphanCompanions() throws {
    let missingPath = try writeSyntheticSafetensors([
        .init(name: "proj.weight", dtype: "U32", shape: [1, 1],
              bytes: Data([0, 0, 0, 0])),
        .init(name: "proj.scales", dtype: "F16", shape: [1, 1],
              bytes: u16Data([0.5])),
    ])
    defer { try? FileManager.default.removeItem(atPath: missingPath) }
    #expect(throws: SmeltPrequantizedTensorError.self) {
        try SmeltPrequantizedSafetensors(
            loader: SafetensorsLoader(paths: [missingPath]),
            format: .binary1, groupSize: 32)
    }

    let orphanPath = try writeSyntheticSafetensors([
        .init(name: "orphan.scales", dtype: "F16", shape: [1, 1],
              bytes: u16Data([0.5])),
    ])
    defer { try? FileManager.default.removeItem(atPath: orphanPath) }
    #expect(throws: SmeltPrequantizedTensorError.self) {
        try SmeltPrequantizedSafetensors(
            loader: SafetensorsLoader(paths: [orphanPath]),
            format: .binary1, groupSize: 32)
    }
}

@Test func nativeWeightWriterStreamsCanonicalSignedAndDenseRegions() throws {
    let sourcePath = try writeSyntheticSafetensors([
        .init(name: "model.proj.weight", dtype: "U32", shape: [1, 1],
              bytes: Data([0x55, 0x55, 0x55, 0x55])),
        .init(name: "model.proj.scales", dtype: "F16", shape: [1, 1],
              bytes: u16Data([0.5])),
        .init(name: "model.proj.biases", dtype: "F16", shape: [1, 1],
              bytes: u16Data([-0.25])),
        .init(name: "model.norm.weight", dtype: "F32", shape: [2],
              bytes: f32Data([1.5, -2.0])),
    ])
    let outputPath = NSTemporaryDirectory()
        + "native_signed_\(ProcessInfo.processInfo.globallyUniqueString).bin"
    defer {
        try? FileManager.default.removeItem(atPath: sourcePath)
        try? FileManager.default.removeItem(atPath: outputPath)
    }

    let source = try SmeltPrequantizedSafetensors(
        loader: SafetensorsLoader(paths: [sourcePath]), format: .binary1, groupSize: 32)
    let packed = try #require(source.prequantizedTensors.first)
    let normDescriptor = try #require(source.checkpointTensors.first)
    let layout = [
        SmeltWeightEntry(
            name: "norm_weight", offset: 0, sizeBytes: 4,
            shape: [2], dtype: .fp16),
        SmeltWeightEntry(
            name: "proj_weight", offset: 128, sizeBytes: 4,
            shape: [1, 32], dtype: .binary1, groupSize: 32,
            packedRowStride: 4, paddedCols: 32,
            scalesOffset: 256, scalesSizeBytes: 2),
    ]

    _ = try SmeltNativeWeightWriter.write(
        packedTensors: [.init(runtimeName: "proj_weight", view: packed)],
        denseTensors: [(
            runtimeName: "norm_weight",
            data: source.checkpointTensorData(normDescriptor),
            byteCount: normDescriptor.byteCount,
            shape: normDescriptor.shape,
            dtype: normDescriptor.dtype
        )],
        expectedLayout: layout,
        outputPath: outputPath
    )

    let bytes = try Data(contentsOf: URL(fileURLWithPath: outputPath))
    #expect(bytes.count == 258)
    try bytes.withUnsafeBytes { raw in
        #expect(Float16(bitPattern: raw.loadUnaligned(as: UInt16.self)) == 1.5)
        #expect(Float16(bitPattern: raw.loadUnaligned(fromByteOffset: 2, as: UInt16.self)) == -2)
        #expect(Array(bytes[128..<132]) == [0x55, 0x55, 0x55, 0x55])
        #expect(Float16(bitPattern: raw.loadUnaligned(
            fromByteOffset: 256, as: UInt16.self)) == 0.25)

        var dequantized = [Float](repeating: 0, count: 32)
        try dequantized.withUnsafeMutableBufferPointer { output in
            try SmeltSignedQuantCodec.dequantizeRow(
                format: .binary1,
                codes: raw.baseAddress!.advanced(by: 128),
                codeByteCount: 4,
                scales: raw.baseAddress!.advanced(by: 256),
                scaleByteCount: 2,
                cols: 32,
                paddedCols: 32,
                groupSize: 32,
                into: output.baseAddress!
            )
        }
        #expect(dequantized[0] == 0.25)
        #expect(dequantized[1] == -0.25)
    }
}

@Test func nativeWeightWriterPreservesCanonicalInterleavedTernaryCodes() throws {
    var interleaved = Data(repeating: 0, count: 8)
    for col in 0..<32 {
        let code: UInt8 = col.isMultiple(of: 3) ? 0 : (col % 3 == 1 ? 1 : 2)
        interleaved[col / 4] |= code << UInt8((col & 3) * 2)
    }
    let sourcePath = try writeSyntheticSafetensors([
        .init(name: "model.proj.weight", dtype: "U32", shape: [1, 2],
              bytes: interleaved),
        .init(name: "model.proj.scales", dtype: "F16", shape: [1, 1],
              bytes: u16Data([0.5])),
        .init(name: "model.proj.biases", dtype: "F16", shape: [1, 1],
              bytes: u16Data([-0.5])),
    ])
    let outputPath = NSTemporaryDirectory()
        + "native_ternary_\(ProcessInfo.processInfo.globallyUniqueString).bin"
    defer {
        try? FileManager.default.removeItem(atPath: sourcePath)
        try? FileManager.default.removeItem(atPath: outputPath)
    }

    let source = try SmeltPrequantizedSafetensors(
        loader: SafetensorsLoader(paths: [sourcePath]), format: .ternary2, groupSize: 32)
    let packed = try #require(source.prequantizedTensors.first)
    let layout = [
        SmeltWeightEntry(
            name: "proj_weight", offset: 0, sizeBytes: 8,
            shape: [1, 32], dtype: .ternary2, groupSize: 32,
            packedRowStride: 8, paddedCols: 32,
            scalesOffset: 128, scalesSizeBytes: 2),
    ]
    _ = try SmeltNativeWeightWriter.write(
        packedTensors: [.init(runtimeName: "proj_weight", view: packed)],
        denseTensors: [], expectedLayout: layout, outputPath: outputPath)

    let bytes = try Data(contentsOf: URL(fileURLWithPath: outputPath))
    try bytes.withUnsafeBytes { raw in
        #expect(Data(raw.prefix(8)) == interleaved)
        #expect(Float16(bitPattern: raw.loadUnaligned(
            fromByteOffset: 128, as: UInt16.self)) == 0.5)

        var dequantized = [Float](repeating: 0, count: 32)
        try dequantized.withUnsafeMutableBufferPointer { output in
            try SmeltSignedQuantCodec.dequantizeRow(
                format: .ternary2,
                codes: raw.baseAddress!,
                codeByteCount: 8,
                scales: raw.baseAddress!.advanced(by: 128),
                scaleByteCount: 2,
                cols: 32,
                paddedCols: 32,
                groupSize: 32,
                into: output.baseAddress!
            )
        }
        for col in 0..<32 {
            #expect(dequantized[col] == Float(col % 3 - 1) * 0.5)
        }
    }
}
